# A  python demo run JunoMWR forward RT using Canoe

## Install and compile

Clone to local directory and check into branch `jh/juno_mwr_fwd`:
```bash
local $ git clone https://github.com/jihenghu/canoe.git
local $ cd canoe
local/canoe $ git checkout jh/juno_mwr_fwd
local/canoe $ cd ..

```

make a build path and compile:
```bash
local $ mkdir build
local $ cd build
local/build $ cmake ../canoe -DTASK=juno
......

local/build $ make -j8
```

## Demos
There are three main demo files under `build/bin/`:
- `run_juno_forward_mwr.py` : a simplest demo, RT in Jovian Equtorial Zone, with
	- `moist adiabatic` model, with a fixed 1-bar temperature;
	- `constant NH3 abundace` in Jovian Equtorial Zone;
	- parameters are taken from [Cheng (2020)](https://www.nature.com/articles/s41550-020-1009-3);
	- key configurations are specfied in `juno_mwr.yaml` and `juno_mwr.inp`.
	- output profiles of NH3, H2O and Temperature;
	- as well as radiances:
		![radiance_dry.png](radiance_dry.png)
	
- `demo_juno_mwr_fwd_EZ_fix_T1bar.ipynb`: a test comparing the dry vs. moist adiabatic modelings, with the same T1bar.
	![Dry vs. moist (Fixed 1bar)](temp_profile_fixt1bar.png)

- `demo_juno_mwr_fwd_EZ_fix_Ts.ipynb`: a test comparing the dry vs. moist adiabatic modelings, with the same bottom temperature.
	![Dry vs. moist (Fixed bottom)](temp_profile_fixtts.png)
		





