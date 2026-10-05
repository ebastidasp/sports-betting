import unittest
import numpy as np
from scipy.stats import poisson
import football_poisson as m

class ProbabilityTests(unittest.TestCase):
    def test_against_large_score_matrix(self):
        for h,a in ((1.3,0.8),(8,7),(0.05,8)):
            for rho in (0,-0.08,0.08):
                goals=np.arange(100)
                matrix=np.outer(poisson.pmf(goals,h),poisson.pmf(goals,a))
                r=np.clip(rho,max(-1/h,-1/a)+1e-9,min(1,1/(h*a))-1e-9)
                matrix[0,0]*=1-h*a*r
                matrix[0,1]*=1+h*r
                matrix[1,0]*=1+a*r
                matrix[1,1]*=1-r
                expected=[np.tril(matrix,-1).sum(),np.trace(matrix),np.triu(matrix,1).sum()]
                np.testing.assert_allclose(m.outcome_probabilities(h,a,rho=rho),expected,atol=1e-12)
    def test_invalid_rates(self):
        for rate in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                m.outcome_probabilities(rate,1)
    def test_symmetry(self):
        np.testing.assert_allclose(m.outcome_probabilities(2,1,rho=-.1),
                                   m.outcome_probabilities(1,2,rho=-.1)[::-1])

class SelectionTests(unittest.TestCase):
    def test_chronological_folds_and_baseline_tie(self):
        import pandas as pd
        from unittest.mock import patch
        dates=pd.date_range("2020-01-01", periods=400, tz="UTC")
        data=pd.DataFrame({"kickoff":dates,"fixture_id":range(400),"league_id":1,
                           "home_goals":1,"away_goals":0})
        boundaries=[]
        def fit(history, cutoff, *parameters):
            self.assertTrue((history.kickoff < cutoff).all())
            boundaries.append(cutoff)
            return {},0
        def coverage(models, validation):
            self.assertTrue((validation.kickoff >= boundaries[-1]).all())
            return validation
        with patch.object(m,"fit_team_bundle",side_effect=fit), \
             patch.object(m,"supported_fixtures",side_effect=coverage), \
             patch.object(m,"predict_team_rates",side_effect=lambda models,v:(np.ones(len(v)),np.ones(len(v)))):
            selected,report=m.select_parameters(data,.01,.95,True,1)
        self.assertEqual(selected,(.01,.95,True))
        self.assertEqual([f["matches"] for f in report["folds"]],[80,80])

if __name__=='__main__': unittest.main()
