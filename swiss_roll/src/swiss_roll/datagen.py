import numpy as np
import argparse 
import torch 
from sklearn.datasets import make_swiss_roll
from swiss_roll import DATA_DIR
np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--samples', type=int, default = 100_000, required=False)
parser.add_argument('--noise', type=float, default = 0.5 ,required=False)

def normalise(array: np.array): 
    shifted = array - np.mean(array)
    scaled = shifted / np.std(array) 
    return scaled 

def normalised_swiss_roll(samps: int, noise: np.float32) -> tuple[np.array, np.array]:
    xyz_points, manifold_points = make_swiss_roll(n_samples=samps, noise=noise) 
    xyz_points = normalise(xyz_points)
    manifold_points = normalise(manifold_points)
    return xyz_points, manifold_points

def writeout_data(data: np.array):
    writeout_path = DATA_DIR / "swiss_roll.npz"
    with open(writeout_path, 'wb') as f: 
        np.savez(f, **data)

def load_swissroll(path=None):
    if not path: 
        path = DATA_DIR / 'swiss_roll.npz'
       
    with open(path, 'rb') as f: 
        data = np.load(f)
        xyz_points = data['xyz_points']
        manifold_points = data['manifold_points']
     
    return torch.tensor(xyz_points, dtype=torch.float32), torch.tensor(manifold_points, dtype=torch.float32).view(-1, 1)

if __name__ == '__main__':
    args = parser.parse_args()
    xyz_points, manifold_points = normalised_swiss_roll(args.samples, args.noise)
    data = {
        'xyz_points': xyz_points,
        'manifold_points': manifold_points
    }
    writeout_data(data)