"""Frozen input/output equivalence, not a scientific or closed-loop evaluation."""
from pathlib import Path
import sys,json,unittest,importlib,warnings
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from handoff import ControlGMM1,GMM2,PaperGMM1,history_features

class HandoffTests(unittest.TestCase):
    def test_models_and_history(self):
        g1=ControlGMM1();models={k:GMM2(k) for k in ('current','history')}
        for sid in json.loads((ROOT/'examples/index.json').read_text()):
            with self.subTest(scene=sid):
                p=ROOT/'examples'/sid;d=json.loads((p/'input.json').read_text());e=json.loads((p/'expected.json').read_text())
                history=history_features(pd.read_csv(p/'history.csv'))
                np.testing.assert_allclose(list(history.values()),list(d['history'].values()),rtol=1e-9,atol=1e-10)
                pred=g1.predict(**d['gmm1'])
                np.testing.assert_allclose(pred.gmr_features,e['gmm1_features'],rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(pred.mu_deltaY,e['gmm1_mean'],rtol=1e-9,atol=1e-10)
                np.testing.assert_allclose(pred.Sigma_deltaY,e['gmm1_covariance'],rtol=1e-9,atol=1e-10)
                for variant,model in models.items():
                    out=model.predict({**d['current'],**history});expected=e['gmm2_'+variant]
                    for key in ['mean','covariance']:
                        np.testing.assert_allclose(out[key],expected[key],rtol=1e-8,atol=1e-9)
                    np.testing.assert_allclose(out['next_vr'][0],expected['next_vr'],rtol=1e-8,atol=1e-9)
    def test_f1_corrected_qp_reference(self):
        from future_work.short_horizon_output_residual.gmm1_foundation_stage1.model import FoundationAdapter,FoundationBridge
        from future_work.short_horizon_output_residual.ft03_variable_speed_nominal_validation.variable_speed_mpc import VariableSpeedNominalController,SpeedProfile
        from future_work.short_horizon_output_residual.mpc_integration.rollout_aligned_sequential_layer import RolloutAlignedSequentialLayer
        profile=SpeedProfile(np.array([0.,1000.]),np.array([30.,30.]),'handoff')
        for sid in json.loads((ROOT/'examples/index.json').read_text()):
            p=ROOT/'examples'/sid;d=json.loads((p/'input.json').read_text());e=json.loads((p/'expected.json').read_text())['f1_adapter']
            controller=VariableSpeedNominalController(profile)
            adapter=FoundationAdapter(controller.optimizer.config,layer=RolloutAlignedSequentialLayer(),max_iterations=1)
            _,u=adapter.iterate_variable(d['gmm1']['state'],np.asarray(d['error_state']),np.asarray(d['curvature']),np.asarray(d['gmm1']['vx_preview']),d['gmm1']['condition']['steering'],np.asarray(d['gmm1']['controls']))
            np.testing.assert_allclose(u,e['u1'],rtol=1e-7,atol=1e-8)
            np.testing.assert_allclose(adapter.selected_prediction,e['states'],rtol=1e-7,atol=1e-8)
    def test_paper_separate_interface(self):
        d=json.loads((ROOT/'examples/paper_k15.json').read_text())
        p=PaperGMM1().predict(d['x'])
        np.testing.assert_allclose(p,d['expected_mean'],rtol=1e-8,atol=1e-9)
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):GMM2('wrong')
        with self.assertRaises(KeyError):GMM2('current').predict({})
        with self.assertRaises(ValueError):history_features(pd.DataFrame())
        with self.assertRaises(ValueError):ControlGMM1().predict({'vy':0.,'r':0.},{'vx':20.,'steering':0.},[0.])
    def test_project_imports_are_local(self):
        for name,module in list(sys.modules.items()):
            if name.split('.')[0] in ('handoff','future_work','residual_model','bicycle_model'):
                path=getattr(module,'__file__',None)
                if path:self.assertTrue(Path(path).resolve().is_relative_to(ROOT/'src'),(name,path))

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HandoffTests))
    if '--report' in sys.argv:
        (ROOT/'verification_result.json').write_text(json.dumps({'passed':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'python':sys.version,'directory':str(ROOT),'scope':'inference and F1 corrected-QP reference replay; no Chrono closed-loop run'},indent=2),encoding='utf-8')
    sys.exit(0 if result.wasSuccessful() else 1)
