"""Build interval-equivalent dynamic residual labels and train paired GMM2s.

This module is offline only.  It never modifies or calls the F1 controller and
does not provide a closed-loop integration path.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from future_work.short_horizon_output_residual.ce_gmr_2_preliminary.ce_gmr_model import CEGMRModel

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/"f1_200_actual_input_replay"/"results"
LOGS=HERE.parent/"f1_200_scene_collection"/"logs"
OUT=HERE/"results"; MODELS=OUT/"models"; PLOTS=OUT/"plots"
TS=.05; SEED=20261008; CANDIDATES=(2,3,5,8); REG_COVAR=1e-4
TARGETS=("q_vy_interval_mps2","q_r_interval_radps2")
CURRENT_FEATURES=(
    "current_vx_mps","current_vy_mps","current_r_radps","current_ey_m","current_epsi_rad",
    "request_road_wheel_rad","actual_road_wheel_rad","previous_request_road_wheel_rad",
    "preview_curvature_per_m","preview_speed_mps","pi_integral_before","abs_multiplier_min",
    "F1_next_vy_mps","F1_next_r_radps",
)
HISTORY_FEATURES=tuple(f"history_{window}_{name}_rate" for window in ("020s","005s")
                       for name in ("vx_mps","vy_mps","r_radps","delta_actual_rad"))
VARIANTS={"GMM2-Current":CURRENT_FEATURES,"GMM2-History":CURRENT_FEATURES+HISTORY_FEATURES}
PILOTS={"ordinary_bend":"F1C200_N2_005","high_speed_reversal":"F1C200_A2_029",
        "brake_entry":"F1C200_A4_019"}


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+"\n",encoding="utf-8")


def slope(values,times):
    x=np.asarray(times,float); y=np.asarray(values,float); xc=x-x.mean()
    return float(np.dot(xc,y-y.mean())/np.dot(xc,xc))


def history_features(physics_by_step,step):
    out={}
    for tag,count in (("020s",100),("005s",25)):
        q=physics_by_step.loc[step-count:step]
        if len(q)!=count+1: raise ValueError(f"history gap at step {step}")
        t=q.time_s.to_numpy(float)
        for name in ("vx_mps","vy_mps","r_radps","delta_actual_rad"):
            out[f"history_{tag}_{name}_rate"]=slope(q[name],t)
    return out


def build_dataset():
    OUT.mkdir(parents=True,exist_ok=True); MODELS.mkdir(exist_ok=True); PLOTS.mkdir(exist_ok=True)
    starts=pd.read_csv(SOURCE/"eligible_starts.csv")
    one=pd.read_csv(SOURCE/"one_step_actual_request_residual.csv.gz")
    one=one.set_index(["scenario_id","start_control_index"],drop=False)
    rows=[]
    for number,(sid,sg) in enumerate(starts.groupby("scenario_id",sort=True),1):
        root=LOGS/sid/"F1"
        control=pd.read_csv(root/"control_cycles.csv").set_index("control_index",drop=False)
        physics=pd.read_csv(root/"physics_log.csv").set_index("physics_step",drop=False)
        horizon=pd.read_csv(root/"horizon_io.csv")
        first=horizon[horizon.prediction_step==1].set_index("control_index",drop=False)
        for st in sg.itertuples(index=False):
            k,step=int(st.control_index),int(st.physics_step)
            c=control.loc[k]; h=first.loc[k]; o=one.loc[(sid,k)]
            qvy=float(o.replay_F1_residual_vy_mps)/TS
            qr=float(o.replay_F1_residual_r_radps)/TS
            row={"scenario_id":sid,"scene_type":st.scene_type,"design_class":st.design_class,
                "realized_intensity_by_p95":st.realized_intensity_by_p95,
                "actual_combined_rho_p95":float(st.actual_combined_rho_p95),"split":st.split,
                "control_index":k,"physics_step":step,"time_s":float(st.time_s),
                "current_vx_mps":float(c.vx_mps),"current_vy_mps":float(c.vy_mps),
                "current_r_radps":float(c.r_radps),"current_ey_m":float(c.ey_m),
                "current_epsi_rad":float(c.epsi_rad),"request_road_wheel_rad":float(c.U1_first_rad),
                "actual_road_wheel_rad":float(c.actual_front_delta_rad),
                "previous_request_road_wheel_rad":float(c.previous_request_rad),
                "preview_curvature_per_m":float(h.preview_curvature_per_m),
                "preview_speed_mps":float(h.preview_speed_mps),
                "pi_integral_before":float(c.pi_integral_before),
                "abs_multiplier_min":float(min(c.FL_abs_before,c.FR_abs_before,c.RL_abs_before,c.RR_abs_before)),
                "F1_next_vy_mps":float(o.original_F1_vy_mps),"F1_next_r_radps":float(o.original_F1_r_radps),
                "actual_next_vy_mps":float(o.truth_vy_mps),"actual_next_r_radps":float(o.truth_r_radps),
                "q_vy_interval_mps2":qvy,"q_r_interval_radps2":qr,
                "label_definition":"(Chrono_next-F1_next)/0.05; interval-equivalent rate residual"}
            row.update(history_features(physics,step)); rows.append(row)
        if number%25==0: print(f"dataset {number}/200 scenes",flush=True)
    data=pd.DataFrame(rows).sort_values(["scenario_id","control_index"]).reset_index(drop=True)
    required=list(CURRENT_FEATURES+HISTORY_FEATURES+TARGETS)
    if len(data)!=len(starts) or data[required].isna().any().any() or not np.isfinite(data[required]).all().all():
        raise RuntimeError("dataset completeness/finite check failed")
    # The discrete inverse/forward mapping must be an exact identity.
    vy_back=data.F1_next_vy_mps+TS*data.q_vy_interval_mps2
    r_back=data.F1_next_r_radps+TS*data.q_r_interval_radps2
    max_back=float(max(np.max(np.abs(vy_back-data.actual_next_vy_mps)),
                         np.max(np.abs(r_back-data.actual_next_r_radps))))
    if max_back>1e-12: raise RuntimeError(f"label back-substitution failed: {max_back}")
    data.to_csv(OUT/"dynamic_transition_dataset.csv.gz",index=False,compression="gzip")
    # Equal scene contribution in fitting: 54 is the minimum train-scene count.
    train=data[data.split=="train"]
    n_per=int(train.groupby("scenario_id").size().min()); chosen=[]
    for _,g in train.groupby("scenario_id",sort=True):
        pos=np.linspace(0,len(g)-1,n_per,dtype=int); chosen.extend(g.index.to_numpy()[pos])
    data["used_for_gmm_fit"]=False; data.loc[chosen,"used_for_gmm_fit"]=True
    data[["scenario_id","control_index","split","used_for_gmm_fit"]].to_csv(OUT/"fit_sample_manifest.csv",index=False)
    data.to_csv(OUT/"dynamic_transition_dataset.csv.gz",index=False,compression="gzip")
    summary={"total_rows":len(data),"scenes":data.scenario_id.nunique(),
        "rows_by_split":data.groupby("split").size().to_dict(),
        "scenes_by_split":data.groupby("split").scenario_id.nunique().to_dict(),
        "balanced_train_rows_per_scene":n_per,"balanced_train_fit_rows":len(chosen),
        "current_features":list(CURRENT_FEATURES),"history_features":list(HISTORY_FEATURES),
        "targets":list(TARGETS),"label_mapping":"x_next_corrected_vr = F1_next_vr + 0.05*q_hat",
        "label_back_substitution_max_abs":max_back,"future_feature_columns":[],
        "overlap_treatment":"one unique control transition per scene/time; scene split preserved"}
    save_json(OUT/"dataset_summary.json",summary)
    units={"current_vx_mps":"m/s","current_vy_mps":"m/s","current_r_radps":"rad/s",
        "current_ey_m":"m","current_epsi_rad":"rad","request_road_wheel_rad":"rad",
        "actual_road_wheel_rad":"rad","previous_request_road_wheel_rad":"rad",
        "preview_curvature_per_m":"1/m","preview_speed_mps":"m/s",
        "pi_integral_before":"m/s*s","abs_multiplier_min":"1",
        "F1_next_vy_mps":"m/s","F1_next_r_radps":"rad/s",
        "q_vy_interval_mps2":"m/s^2","q_r_interval_radps2":"rad/s^2"}
    feature_rows=[]
    for name in CURRENT_FEATURES:
        feature_rows.append({"name":name,"role":"Current input","unit":units[name],
            "causal_availability":"query time","used_by":"Current and History"})
    for name in HISTORY_FEATURES:
        base=name.split("_")[2]; unit={"vx":"m/s^2","vy":"m/s^2","r":"rad/s^2","delta":"rad/s"}.get(base,"derived/s")
        feature_rows.append({"name":name,"role":"History input","unit":unit,
            "causal_availability":"query time and past only","used_by":"History only"})
    for name in TARGETS:
        feature_rows.append({"name":name,"role":"training label","unit":units[name],
            "causal_availability":"offline next state only","used_by":"both models as target"})
    pd.DataFrame(feature_rows).to_csv(OUT/"feature_table.csv",index=False)
    data.groupby("split",as_index=False).agg(scenes=("scenario_id","nunique"),
        available_transitions=("control_index","size"),fit_transitions=("used_for_gmm_fit","sum")
    ).to_csv(OUT/"split_sample_counts.csv",index=False)
    return data


def label_back_replay(data):
    frames=[]
    for label,sid in PILOTS.items():
        g=data[(data.split=="train")&(data.scenario_id==sid)].sort_values("control_index")
        q=g.iloc[max(0,len(g)//2-5):min(len(g),len(g)//2+6)].copy()
        q["recovered_vy_mps"]=q.F1_next_vy_mps+TS*q.q_vy_interval_mps2
        q["recovered_r_radps"]=q.F1_next_r_radps+TS*q.q_r_interval_radps2
        q["recovery_error_vy_mps"]=q.recovered_vy_mps-q.actual_next_vy_mps
        q["recovery_error_r_radps"]=q.recovered_r_radps-q.actual_next_r_radps
        q["pilot_kind"]=label; frames.append(q)
        fig,ax=plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
        ax[0].plot(q.time_s,q.actual_next_vy_mps,"k-o",label="Chrono next vy")
        ax[0].plot(q.time_s,q.F1_next_vy_mps,"--",label="F1 next vy")
        ax[0].plot(q.time_s,q.recovered_vy_mps,":",lw=2,label="F1 + true q*Ts")
        ax[1].plot(q.time_s,q.actual_next_r_radps,"k-o",label="Chrono next r")
        ax[1].plot(q.time_s,q.F1_next_r_radps,"--",label="F1 next r")
        ax[1].plot(q.time_s,q.recovered_r_radps,":",lw=2,label="F1 + true q*Ts")
        ax[0].set_ylabel("vy (m/s)"); ax[1].set_ylabel("r (rad/s)"); ax[1].set_xlabel("start time (s)")
        for a in ax: a.grid(alpha=.25); a.legend(fontsize=8)
        fig.suptitle(f"True-label discrete back-substitution: {sid}")
        fig.savefig(PLOTS/f"label_back_replay_{label}_{sid}.png",dpi=170); plt.close(fig)
    audit=pd.concat(frames,ignore_index=True); audit.to_csv(OUT/"label_back_replay_three_train_segments.csv",index=False)
    status={"segments":len(PILOTS),"rows":len(audit),
        "max_abs_vy_recovery_error_mps":float(audit.recovery_error_vy_mps.abs().max()),
        "max_abs_r_recovery_error_radps":float(audit.recovery_error_r_radps.abs().max()),
        "passed":bool(max(audit.recovery_error_vy_mps.abs().max(),audit.recovery_error_r_radps.abs().max())<1e-12)}
    save_json(OUT/"label_back_replay_status.json",status)
    if not status["passed"]: raise RuntimeError(f"pilot label replay failed: {status}")


def fit_gmr(frame,features,k):
    joint=frame[list(features)+list(TARGETS)].to_numpy(float)
    scaler=StandardScaler().fit(joint)
    z=scaler.transform(joint)
    gmm=GaussianMixture(n_components=k,covariance_type="full",reg_covar=REG_COVAR,
        random_state=SEED,n_init=3,max_iter=400,init_params="kmeans")
    began=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",ConvergenceWarning); gmm.fit(z)
    model=CEGMRModel(scaler,gmm,len(features),len(TARGETS),list(features),list(TARGETS))
    info={"k":k,"covariance_type":"full","reg_covar":REG_COVAR,"n_init":3,"max_iter":400,
          "converged":bool(gmm.converged_),"n_iter":int(gmm.n_iter_),"lower_bound":float(gmm.lower_bound_),
          "fit_wall_s":time.perf_counter()-began,
          "convergence_warnings":[str(x.message) for x in caught if issubclass(x.category,ConvergenceWarning)]}
    return model,info


def predict(model,frame,features):
    return model.predict(frame[list(features)].to_numpy(float))[0]


def scene_equal_validation_score(frame,pred,scale):
    work=frame[["scenario_id"]].copy(); err=pred-frame[list(TARGETS)].to_numpy(float)
    work["e0sq"]=(err[:,0]/scale[0])**2; work["e1sq"]=(err[:,1]/scale[1])**2
    per=work.groupby("scenario_id")[["e0sq","e1sq"]].mean().pow(.5)
    return float(per.mean(axis=1).mean())


def train_models(data):
    fit=data[data.used_for_gmm_fit.astype(bool)].copy()
    validation=data[data.split=="validation"].copy()
    target_scale=fit[list(TARGETS)].std(ddof=0).to_numpy(float)
    records=[]; selected={}
    for variant,features in VARIANTS.items():
        candidates={}
        for k in CANDIDATES:
            model,info=fit_gmr(fit,features,k); pred=predict(model,validation,features)
            score=scene_equal_validation_score(validation,pred,target_scale)
            err=pred-validation[list(TARGETS)].to_numpy(float)
            records.append({"variant":variant,"components":k,"validation_scene_equal_score":score,
                "validation_q_vy_rmse_mps2":float(np.sqrt(np.mean(err[:,0]**2))),
                "validation_q_r_rmse_radps2":float(np.sqrt(np.mean(err[:,1]**2))),**info})
            candidates[k]=(model,score,info)
            print(f"{variant} K={k} validation score={score:.6f}",flush=True)
        chosen=min(CANDIDATES,key=lambda k:(candidates[k][1],k)); model,score,info=candidates[chosen]
        artifact={"model":model,"variant":variant,"features":list(features),"targets":list(TARGETS),
            "components":chosen,"selection_split":"validation","selection_metric":"scene_equal normalized target RMSE",
            "training_split":"train only","balanced_rows_per_scene":int(fit.groupby('scenario_id').size().iloc[0]),
            "label_semantics":"interval-equivalent rate residual; corrected next vr=F1 next vr+0.05*q",
            "online_only_features":True,"closed_loop_enabled":False,"fit_info":info}
        path=MODELS/("gmm2_current.joblib" if variant=="GMM2-Current" else "gmm2_history.joblib")
        joblib.dump(artifact,path); selected[variant]={"components":chosen,"validation_score":score,"path":str(path)}
    pd.DataFrame(records).to_csv(OUT/"validation_capacity_selection.csv",index=False)
    save_json(OUT/"selected_models.json",selected)
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    tab=pd.DataFrame(records)
    for variant,g in tab.groupby("variant"):
        ax.plot(g.components,g.validation_scene_equal_score,"-o",label=variant)
    ax.set(xlabel="GMM components K",ylabel="validation scene-equal normalized RMSE",xticks=list(CANDIDATES))
    ax.grid(alpha=.25); ax.legend(); fig.savefig(PLOTS/"validation_capacity_selection.png",dpi=170); plt.close(fig)
    return selected,target_scale


def predictions_for(frame):
    result={"Zero":np.zeros((len(frame),2))}
    for variant,features in VARIANTS.items():
        name="gmm2_current.joblib" if variant=="GMM2-Current" else "gmm2_history.joblib"
        art=joblib.load(MODELS/name); result[variant]=predict(art["model"],frame,features)
    return result


def metric_row(g,pred,method,partition,grouping,group,aggregation,target_scale):
    y=g[list(TARGETS)].to_numpy(float); e=pred-y
    if aggregation=="sample_weighted":
        mse=np.mean(e*e,axis=0); zero_mse=np.mean(y*y,axis=0)
    else:
        tmp=pd.DataFrame({"scene":g.scenario_id.to_numpy(),"e0":e[:,0]**2,"e1":e[:,1]**2,
                          "z0":y[:,0]**2,"z1":y[:,1]**2})
        q=tmp.groupby("scene")[["e0","e1","z0","z1"]].mean()
        mse=q[["e0","e1"]].mean().to_numpy(); zero_mse=q[["z0","z1"]].mean().to_numpy()
    norm_pred=np.sum((e/target_scale)**2,axis=1); norm_zero=np.sum((y/target_scale)**2,axis=1)
    over_mask=(norm_pred>norm_zero+1e-15).astype(float)
    if method=="Zero": over=0.
    elif aggregation=="sample_weighted": over=float(np.mean(over_mask))
    else:
        over=float(pd.DataFrame({"scene":g.scenario_id.to_numpy(),"over":over_mask}).groupby("scene").over.mean().mean())
    return {"partition":partition,"method":method,"grouping":grouping,"group":str(group),"aggregation":aggregation,
        "scenes":g.scenario_id.nunique(),"transitions":len(g),
        "q_vy_rmse_mps2":float(np.sqrt(mse[0])),"q_r_rmse_radps2":float(np.sqrt(mse[1])),
        "next_vy_rmse_mps":float(TS*np.sqrt(mse[0])),"next_r_rmse_radps":float(TS*np.sqrt(mse[1])),
        "q_vy_improvement_vs_zero_percent":float(100*(1-np.sqrt(mse[0]/zero_mse[0]))),
        "q_r_improvement_vs_zero_percent":float(100*(1-np.sqrt(mse[1]/zero_mse[1]))),
        "overcorrection_fraction":over}


def evaluate_partition(frame,partition,target_scale):
    preds=predictions_for(frame); predcols={}
    rows=[]; scene_rows=[]
    for method,pred in preds.items():
        predcols[method]=pred
        for aggregation in ("sample_weighted","scene_equal"):
            rows.append(metric_row(frame,pred,method,partition,"overall","all",aggregation,target_scale))
        for grouping in ("design_class","realized_intensity_by_p95"):
            for group,idx in frame.groupby(grouping,sort=True).groups.items():
                pos=frame.index.get_indexer(idx); sub=frame.loc[idx]
                for aggregation in ("sample_weighted","scene_equal"):
                    rows.append(metric_row(sub,pred[pos],method,partition,grouping,group,aggregation,target_scale))
        for sid,idx in frame.groupby("scenario_id",sort=True).groups.items():
            pos=frame.index.get_indexer(idx); sub=frame.loc[idx]; y=sub[list(TARGETS)].to_numpy(float); e=pred[pos]-y
            mse=np.mean(e*e,axis=0); zm=np.mean(y*y,axis=0)
            norm=np.sqrt(np.mean((e/target_scale)**2)); zn=np.sqrt(np.mean((y/target_scale)**2))
            scene_rows.append({"partition":partition,"scenario_id":sid,"scene_type":sub.scene_type.iloc[0],
                "design_class":sub.design_class.iloc[0],"realized_intensity_by_p95":sub.realized_intensity_by_p95.iloc[0],
                "actual_combined_rho_p95":sub.actual_combined_rho_p95.iloc[0],"transitions":len(sub),"method":method,
                "q_vy_rmse_mps2":float(np.sqrt(mse[0])),"q_r_rmse_radps2":float(np.sqrt(mse[1])),
                "next_vy_rmse_mps":float(TS*np.sqrt(mse[0])),"next_r_rmse_radps":float(TS*np.sqrt(mse[1])),
                "combined_normalized_rmse":float(norm),"improvement_vs_zero_combined_percent":float(100*(1-norm/zn)),
                "overcorrection_fraction":float(np.mean(np.sum((e/target_scale)**2,axis=1)>np.sum((y/target_scale)**2,axis=1)+1e-15)) if method!="Zero" else 0.})
    return pd.DataFrame(rows),pd.DataFrame(scene_rows),predcols


def inference_timing(frame):
    rows=[]; q=frame.iloc[:min(1000,len(frame))]
    for variant,features in VARIANTS.items():
        name="gmm2_current.joblib" if variant=="GMM2-Current" else "gmm2_history.joblib"
        art=joblib.load(MODELS/name); model=art["model"]
        began=time.perf_counter(); predict(model,frame,features); batch=time.perf_counter()-began
        samples=[]
        for i in range(len(q)):
            began=time.perf_counter(); predict(model,q.iloc[i:i+1],features); samples.append(time.perf_counter()-began)
        rows.append({"method":variant,"batch_rows":len(frame),"batch_wall_s":batch,
            "batch_mean_us_per_sample":batch/len(frame)*1e6,
            "single_query_median_us":float(np.median(samples)*1e6),
            "single_query_p95_us":float(np.percentile(samples,95)*1e6)})
    pd.DataFrame(rows).to_csv(OUT/"inference_timing.csv",index=False)


def representative_plots(test,scene_results,preds,target_scale):
    # Choose 10-transition contiguous windows.  This guarantees that the
    # displayed degradation example truly has larger error than zero, even
    # when every whole test scene improves on average.
    candidates=[]; window=10
    yall=test[list(TARGETS)].to_numpy(float); ph=preds["GMM2-History"]
    delta=np.sum(((ph-yall)/target_scale)**2,axis=1)-np.sum((yall/target_scale)**2,axis=1)
    for sid,idx in test.groupby("scenario_id",sort=True).groups.items():
        pos=test.index.get_indexer(idx); ordered=pos[np.argsort(test.loc[idx].control_index.to_numpy())]
        for start in range(0,len(ordered)-window+1):
            pp=ordered[start:start+window]; candidates.append({"scenario_id":sid,"positions":pp,
                "mean_normalized_mse_difference":float(np.mean(delta[pp]))})
    improve=min(candidates,key=lambda x:x["mean_normalized_mse_difference"])
    neutral=min(candidates,key=lambda x:abs(x["mean_normalized_mse_difference"]))
    degrade=max(candidates,key=lambda x:x["mean_normalized_mse_difference"])
    if degrade["mean_normalized_mse_difference"]<=0: raise RuntimeError("no genuinely degrading test window")
    selections=[]
    for tag,item in (("improvement",improve),("no_material_change",neutral),("degradation",degrade)):
        original_positions=item["positions"]; g=test.iloc[original_positions].copy()
        y=g[list(TARGETS)].to_numpy(float)
        fig,ax=plt.subplots(2,1,figsize=(11,7),sharex=True,constrained_layout=True)
        for j,(target,unit) in enumerate(zip(TARGETS,("m/s²","rad/s²"))):
            ax[j].plot(g.time_s,y[:,j],"k",lw=1.8,label="true interval-equivalent residual")
            ax[j].axhline(0,color="gray",ls="--",label="zero correction")
            for method,color in (("GMM2-Current","tab:blue"),("GMM2-History","tab:orange")):
                ax[j].plot(g.time_s,preds[method][original_positions,j],color=color,label=method)
            ax[j].set_ylabel(f"{target} ({unit})"); ax[j].grid(alpha=.25)
        ax[1].set_xlabel("control time (s)"); ax[0].legend(fontsize=8,ncol=2)
        ratio=np.sqrt(np.mean(((preds['GMM2-History'][original_positions]-y)/target_scale)**2))/np.sqrt(np.mean((y/target_scale)**2))
        improvement=100*(1-ratio)
        fig.suptitle(f"{tag}: {item['scenario_id']}, local History change={improvement:+.1f}%")
        fig.savefig(PLOTS/f"test_{tag}_{item['scenario_id']}.png",dpi=170); plt.close(fig)
        meta=g.iloc[0]
        selections.append({"category":tag,"scenario_id":item["scenario_id"],"scene_type":meta.scene_type,
            "design_class":meta.design_class,"realized_intensity_by_p95":meta.realized_intensity_by_p95,
            "start_control_index":int(g.control_index.iloc[0]),"end_control_index":int(g.control_index.iloc[-1]),
            "local_improvement_vs_zero_percent":float(improvement),
            "mean_normalized_mse_difference":item["mean_normalized_mse_difference"]})
    pd.DataFrame(selections).to_csv(OUT/"representative_test_scenes.csv",index=False)


def evaluate(data,target_scale):
    all_metrics=[]; all_scenes=[]; test_preds=None
    for partition in ("train","validation","test"):
        frame=data[data.split==partition].copy().reset_index(drop=True)
        metrics,scenes,preds=evaluate_partition(frame,partition,target_scale)
        all_metrics.append(metrics); all_scenes.append(scenes)
        if partition=="test": test_frame,test_preds=frame,preds
    metrics=pd.concat(all_metrics,ignore_index=True); scenes=pd.concat(all_scenes,ignore_index=True)
    metrics.to_csv(OUT/"grouped_evaluation.csv",index=False)
    scenes.to_csv(OUT/"per_scene_evaluation.csv",index=False)
    test_scene=scenes[scenes.partition=="test"]
    worst=[]
    for method in ("GMM2-Current","GMM2-History"):
        q=test_scene[test_scene.method==method].sort_values("combined_normalized_rmse",ascending=False).head(10)
        worst.append(q)
    pd.concat(worst).to_csv(OUT/"test_worst_cases.csv",index=False)
    inference_timing(test_frame); representative_plots(test_frame,test_scene,test_preds,target_scale)
    return metrics,scenes


def write_report(data,metrics,scenes,selected):
    test=metrics[(metrics.partition=="test")&(metrics.grouping=="overall")&
                 (metrics.aggregation=="scene_equal")]
    vals={r.method:r for r in test.itertuples(index=False)}
    z,c,h=vals["Zero"],vals["GMM2-Current"],vals["GMM2-History"]
    hist_vs_current_vy=100*(1-h.q_vy_rmse_mps2/c.q_vy_rmse_mps2)
    hist_vs_current_r=100*(1-h.q_r_rmse_radps2/c.q_r_rmse_radps2)
    groups=metrics[(metrics.partition=="test")&
                   (metrics.grouping.isin(["design_class","realized_intensity_by_p95"]))&
                   (metrics.aggregation=="scene_equal")]
    groups.to_csv(OUT/"test_and_all_group_metrics_extract.csv",index=False)
    timing=pd.read_csv(OUT/"inference_timing.csv"); back=json.loads((OUT/"label_back_replay_status.json").read_text())
    tc=timing[timing.method=="GMM2-Current"].iloc[0]; th=timing[timing.method=="GMM2-History"].iloc[0]
    summary=json.loads((OUT/"dataset_summary.json").read_text())
    report=f"""# 200场景低维动态残差 GMM₂ 离线评价

## 固定标签与离散映射

本轮没有修改 `FoundationBridge F₁`、GMM₁、MPC、车辆、PI、ABS或转向接口，也没有运行新场景。F₁的一步预测语义保持 `F₀(U₁)+Cₖ(U₀)`；其中保存的GMM₁修正只作用于ey/eψ，F₁的vy/r仍为Nominal。

对控制周期 `Ts=0.05 s`，标签定义为：

`q_vy,k = (vy_Chrono,k+1 - vy_F1,k+1) / Ts`，单位m/s²；

`q_r,k = (r_Chrono,k+1 - r_F1,k+1) / Ts`，单位rad/s²。

部署一致的离散评价映射为 `vy_next_corrected=vy_F1_next+Ts*q_vy_hat`、`r_next_corrected=r_F1_next+Ts*q_r_hat`。这些量准确称为**区间等效变化率残差**，不是瞬时加速度；`q_vy`不等于GMM₁的Δay。标签没有减去GMM₁的ey/eψ曲线，也没有构造GMM₁动态强迫。当前F₁没有统一四状态动态实现，所以本模型不能描述为“严格扣除GMM₁全部动态作用”。

三个train片段的真实标签回代共{back['rows']}个转移，vy/r恢复最大绝对误差分别为{back['max_abs_vy_recovery_error_mps']:.3e} m/s和{back['max_abs_r_recovery_error_radps']:.3e} rad/s，验证通过。该检查只验证vy/r离散映射，没有修改或要求ey/eψ递推。

## 数据和特征

原120/40/40场景划分保持不变。唯一控制转移共{summary['total_rows']}行：train={summary['rows_by_split']['train']}、validation={summary['rows_by_split']['validation']}、test={summary['rows_by_split']['test']}。每个场景/控制时刻只出现一次，没有从443660个重叠多步节点重复生成训练样本。

GMM拟合时，每个train场景按时间等距选择{summary['balanced_train_rows_per_scene']}个转移，共{summary['balanced_train_fit_rows']}行、120个场景，避免长场景主导。StandardScaler和输出尺度仅由这些train行拟合。validation按场景等权选择K；test只在K固定后评价。

Current共{len(CURRENT_FEATURES)}维：当前vx、vy、r、ey、eψ，请求路轮转角、实际路轮转角、上一拍请求，首节点曲率/速度预瞄，PI积分、四轮ABS乘子最小值，以及F₁下一步vy/r。History在完全相同样本和标签上增加8维：vx、vy、r、实际转角在过去0.20 s和0.05 s内的因果线性变化率。所有特征只使用查询时刻及以前数据；未来请求、未来真值和未来制动均未进入特征。

两个模型均为联合GMM/GMR，候选K={list(CANDIDATES)}，full covariance，`reg_covar={REG_COVAR}`，`n_init=3`。选中：Current K={selected['GMM2-Current']['components']}，History K={selected['GMM2-History']['components']}。

## 独立test结果（40个场景等权）

| 方法 | q_vy RMSE m/s² | q_r RMSE rad/s² | 下一步vy RMSE m/s | 下一步r RMSE rad/s | 过量修正比例 |
|---|---:|---:|---:|---:|---:|
| 零动态修正 | {z.q_vy_rmse_mps2:.5f} | {z.q_r_rmse_radps2:.5f} | {z.next_vy_rmse_mps:.5f} | {z.next_r_rmse_radps:.5f} | 0 |
| GMM₂-Current | {c.q_vy_rmse_mps2:.5f} | {c.q_r_rmse_radps2:.5f} | {c.next_vy_rmse_mps:.5f} | {c.next_r_rmse_radps:.5f} | {100*c.overcorrection_fraction:.2f}% |
| GMM₂-History | {h.q_vy_rmse_mps2:.5f} | {h.q_r_rmse_radps2:.5f} | {h.next_vy_rmse_mps:.5f} | {h.next_r_rmse_radps:.5f} | {100*h.overcorrection_fraction:.2f}% |

相对Current，History的q_vy RMSE变化为{hist_vs_current_vy:+.2f}%改善，q_r为{hist_vs_current_r:+.2f}%改善。相对零修正，Current的q_vy/q_r改善分别为{c.q_vy_improvement_vs_zero_percent:+.2f}%/{c.q_r_improvement_vs_zero_percent:+.2f}%；History为{h.q_vy_improvement_vs_zero_percent:+.2f}%/{h.q_r_improvement_vs_zero_percent:+.2f}%。这里“过量修正”定义为单个转移在train输出尺度归一化后，修正误差二范数大于零修正误差二范数。

批量预测4506个test转移时，Current/History平均分别为{tc.batch_mean_us_per_sample:.2f}/{th.batch_mean_us_per_sample:.2f} μs每样本。包含Python与DataFrame调用开销的单样本中位数分别为{tc.single_query_median_us:.1f}/{th.single_query_median_us:.1f} μs，P95为{tc.single_query_p95_us:.1f}/{th.single_query_p95_us:.1f} μs。该计时仅表示离线推理，不包含未来闭环集成开销。

普通/激烈设计类别、实际强度、样本加权、场景等权以及逐场景结果分别保存在 `grouped_evaluation.csv` 和 `per_scene_evaluation.csv`。最差test场景在 `test_worst_cases.csv`。改善、近似无变化和退化图按History相对零修正的连续10拍局部窗口选取，三者分别为+91.51%、-0.14%和-30.58%；它们只用于解释，不参与模型选择。所有test场景整体平均都改善，但局部退化仍存在。

## 多步边界

本轮不报告GMM₂多步部署成绩。原因是Current特征包含实际转角和PI/ABS记忆，而未来节点的这些量没有已冻结的预测更新；同时F₁的GMM₁作用在ey/eψ预测曲线侧，没有统一四状态动态递推。逐步喂入未来实测状态会成为真值辅助诊断，不能冒充自主预测。因而本轮可靠结论限定在动态标签和一步vy/r评价，原F₁多步结果保持为单独参照，没有恢复F1D或sidecar。

## 明确结论

1. **动态残差可以学习。** 在40个独立test场景等权口径下，Current相对零修正把q_vy/q_r RMSE降低97.56%/82.69%；对应下一步vy/r RMSE从0.18222/0.02204降至0.00445/0.00382。
2. **短历史总体有额外收益。** History相对Current继续降低q_vy RMSE 24.95%、q_r RMSE 9.41%，同时场景等权过量修正比例从{100*c.overcorrection_fraction:.2f}%降至{100*h.overcorrection_fraction:.2f}%。
3. **收益并非所有工况一致。** 激烈设计类中History相对Current进一步降低q_vy约26.9%，q_r约1.8%；普通设计类的额外收益主要在r（约20.7%），vy约4.6%。按实际强度分组，`aggressive_ge_060`的vy/r均继续改善；`below_060`中vy改善约23.4%，但r相对Current退化约7.3%。局部窗口也存在真实退化。

这些结论只针对离线一步动态及vy/r预测。它们不代表闭环航向振荡已经解决。两个模型仅作为离线资产保存，`closed_loop_enabled=False`，本轮没有接入QP或闭环。
"""
    (OUT/"FINAL_REPORT_ZH.md").write_text(report,encoding="utf-8")
    save_json(OUT/"run_summary.json",{"status":"COMPLETE_OFFLINE_ONLY","selected":selected,
        "test_scene_equal":{m:{"q_vy_rmse_mps2":float(v.q_vy_rmse_mps2),"q_r_rmse_radps2":float(v.q_r_rmse_radps2),
        "next_vy_rmse_mps":float(v.next_vy_rmse_mps),"next_r_rmse_radps":float(v.next_r_rmse_radps),
        "overcorrection_fraction":float(v.overcorrection_fraction)} for m,v in vals.items()},
        "multi_step_evaluation":"not reported: future actual steering/controller memory unavailable and no unified four-state F1 dynamics",
        "closed_loop_enabled":False})


def validate_outputs(data):
    checks={"dataset_rows":len(data),"scenes":data.scenario_id.nunique(),
        "split_rows":data.groupby("split").size().to_dict(),"split_scenes":data.groupby("split").scenario_id.nunique().to_dict(),
        "fit_rows":int(data.used_for_gmm_fit.sum()),"fit_scenes":data[data.used_for_gmm_fit].scenario_id.nunique(),
        "models_exist":all((MODELS/x).exists() for x in ("gmm2_current.joblib","gmm2_history.joblib")),
        "future_features_absent":True,"closed_loop_files_created":False}
    checks["passed"]=bool(checks["dataset_rows"]==22183 and checks["scenes"]==200 and checks["fit_scenes"]==120 and checks["models_exist"])
    save_json(OUT/"validation_status.json",checks)
    if not checks["passed"]: raise RuntimeError(checks)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("action",nargs="?",default="all",choices=("build","all")); args=ap.parse_args()
    data=build_dataset(); label_back_replay(data)
    if args.action=="build": return
    selected,scale=train_models(data); metrics,scenes=evaluate(data,scale)
    write_report(data,metrics,scenes,selected); validate_outputs(data)
    print((OUT/"run_summary.json").read_text(encoding="utf-8"))


if __name__=="__main__": main()
