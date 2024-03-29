import numpy as np
rt_arr = np.linspace(0.1,1.0,5) # time of retraining as % of experiment baseline time
ot_arr = np.linspace(0.1,1.0,5) # overhead time per experiment as % of experiment baseline time
from matplotlib import pyplot as plt
from matplotlib.cm import seismic

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    import matplotlib
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap
def compute_cost(rt, ot, n_exp, n_iter):
  total_cost = 0.0
  total_cost += n_iter  # Baseline experiment cost (per iteration)
  total_cost += rt * n_iter # Retraining cost (per iteration)
  total_cost += (n_exp - n_iter) * ot # Sum of the overheads
  return total_cost


timings_diff= [((27.3, 3.9), (35.5,3.8))]
timings_diff2= [((27.3, 3.9), (21. , 8.5))]
j=0

orig_cmap = seismic
#shrunk_cmap = shiftedColorMap(orig_cmap, start=9, midpoint=0.0, stop=-7, name='shrunk')
shrunk_cmap = shiftedColorMap(orig_cmap, midpoint=0.0, name='shrunk')
for n1, n2 in timings_diff2:
  x, y = np.meshgrid(rt_arr, ot_arr)
  tt_arr = (compute_cost(x, y, n1[0], n1[1]) - compute_cost(x, y, n2[0], n2[1]))
  print(tt_arr)
  plt.xlabel("Retraining cost %")
  plt.ylabel("Overhead cost %")
  plt.pcolormesh(rt_arr, ot_arr, tt_arr, cmap='seismic', vmin=-9, vmax=9)
  plt.title('Diff total time (static-dynamic(q0=3))\nas a function of %overhead and %training')
  plt.colorbar()
  #plt.show()
  plt.savefig(f"diff2-q0-3_map.svg")
  j+=1
