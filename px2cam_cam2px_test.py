from cam_math import *
import numpy as np

fabricated_pt = np.array([1,2,3])
fabricated_pt_hat = fabricated_pt / np.linalg.norm(fabricated_pt)

res = np.array([640,360])
fovh_deg = 110
AR = 9/16

px = cam2px(fovh_deg,res,fabricated_pt_hat,AR)
pt_again = px2cam_unit(fovh_deg,res,px,AR)

print(fabricated_pt_hat)
print(pt_again)