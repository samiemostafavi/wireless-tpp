import json
import numpy as np
import pandas as pd

from scipy.interpolate import griddata

def load_data_from_json(json_path):
    """
    Reads the JSON file, returns a dataframe with columns:
       [mcs, rb, p0, p1, p2, p3, pfail].
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    records = []
    for mcs_str, rb_dict in data.items():
        mcs = int(mcs_str)
        for rb_str, stats in rb_dict.items():
            rb = int(rb_str)
            
            retx = stats.get('retx', [0,0,0,0])    # [n0, n1, n2, n3]
            failed = stats.get('failed', 0)
            total = stats.get('total', 0)
            
            if total == 0:
                continue  # skip

            p0 = retx[0] / total
            p1 = retx[1] / total
            p2 = retx[2] / total
            p3 = retx[3] / total
            pf = failed / total
            
            records.append({
                'mcs': mcs,
                'rb': rb,
                'p0': p0,
                'p1': p1,
                'p2': p2,
                'p3': p3,
                'pfail': pf
            })
    
    df = pd.DataFrame(records)
    return df


class GridInterpolationModel:
    """
    A simple class that does 2D interpolation (MCS vs. RB) for
    p0, p1, p2, p3, pfail using SciPy's griddata.
    """
    def __init__(self, df, method='linear'):
        """
        df: DataFrame with columns [mcs, rb, p0, p1, p2, p3, pfail]
        method: 'linear', 'cubic', or 'nearest'
        """
        self.method = method
        
        # Extract the 'points' array of shape (n_samples, 2)
        self.points = df[['mcs','rb']].values.astype(float)
        
        # Extract probability arrays, each shape = (n_samples,)
        self.values_p0 = df['p0'].values
        self.values_p1 = df['p1'].values
        self.values_p2 = df['p2'].values
        self.values_p3 = df['p3'].values
        self.values_pf = df['pfail'].values

    def predict(self, mcs, rb, normalize=True):
        """
        Interpolates [p0, p1, p2, p3, pfail] at a new point (mcs, rb).
        If normalize=True, ensures sum of probabilities = 1 (if not NaN).
        """
        # Prepare (mcs, rb) as shape (1,2)
        query_point = np.array([[mcs, rb]], dtype=float)
        
        # Interpolate each dimension separately
        p0 = griddata(self.points, self.values_p0, query_point, method=self.method)
        p1 = griddata(self.points, self.values_p1, query_point, method=self.method)
        p2 = griddata(self.points, self.values_p2, query_point, method=self.method)
        p3 = griddata(self.points, self.values_p3, query_point, method=self.method)
        pf = griddata(self.points, self.values_pf, query_point, method=self.method)
        
        # Each result is an array of shape (1,) or [nan]. Extract scalar:
        p0, p1, p2, p3, pf = p0[0], p1[0], p2[0], p3[0], pf[0]

        # If all are NaN (outside convex hull, etc.), decide how to handle
        if np.isnan(p0) or np.isnan(p1) or np.isnan(p2) or np.isnan(p3) or np.isnan(pf):
            # e.g. fallback to 0 or some default, or nearest neighbor
            return {
                'p0': None, 'p1': None, 'p2': None, 'p3': None, 'pfail': None
            }
        
        # (Optional) Clip and normalize so sum=1
        # Sometimes interpolation can yield negative or >1, so clip first
        pvals = np.array([p0, p1, p2, p3, pf])
        pvals = np.clip(pvals, 0.0, 1.0)
        if normalize:
            s = pvals.sum()
            if s > 0:
                pvals /= s
        
        return {
            'p0': pvals[0],
            'p1': pvals[1],
            'p2': pvals[2],
            'p3': pvals[3],
            'pfail': pvals[4]
        }


if __name__ == "__main__":
    json_path = "./data/s61-64_results/link_quality/datasets/main_eval/retx_stats.json"  # change to your path
    df = load_data_from_json(json_path)

    # Create the interpolation model
    # method can be 'linear', 'cubic', or 'nearest'
    model = GridInterpolationModel(df, method='linear')
    
    # Example usage:
    # Suppose we want to estimate probabilities at MCS=16, RB=20
    prediction = model.predict(mcs=18, rb=20)
    print(f"Interpolated probabilities: {prediction}")
