# External transfer -- a central finding

Every champion and the seed were applied once, unchanged, to independent datasets (other
sites, amplifiers, populations). In-distribution gains did not reliably transfer.

| task -> external | split | internal Eval seed -> loop | external seed -> loop | ext. gain 95% CI | best-of-20 ext. | Optuna ext. |
|---|---|---|---|---|---|---|
| crc_baxter -> crc_zeller | 1 | 0.854 -> 0.816 | 0.813 -> 0.801 | -0.058 .. +0.035 | 0.797 | 0.813 |
| crc_baxter -> crc_zackular | 1 | 0.854 -> 0.816 | 0.792 -> 0.789 | -0.053 .. +0.053 | 0.737 | 0.792 |
| crc_baxter -> crc_xiang | 1 | 0.854 -> 0.816 | 0.748 -> 0.693 | -0.126 .. -0.007 | 0.679 | 0.748 |
| crc_baxter -> crc_zhao | 1 | 0.854 -> 0.816 | 0.664 -> 0.666 | -0.029 .. +0.031 | 0.680 | 0.664 |
| crc_baxter -> crc_zeller | 2 | 0.756 -> 0.724 | 0.818 -> 0.805 | -0.054 .. +0.024 | 0.818 | 0.830 |
| crc_baxter -> crc_zackular | 2 | 0.756 -> 0.724 | 0.802 -> 0.790 | -0.055 .. +0.034 | 0.802 | 0.802 |
| crc_baxter -> crc_xiang | 2 | 0.756 -> 0.724 | 0.763 -> 0.712 | -0.163 .. +0.054 | 0.763 | 0.758 |
| crc_baxter -> crc_zhao | 2 | 0.756 -> 0.724 | 0.742 -> 0.691 | -0.112 .. +0.006 | 0.742 | 0.743 |
| crc_baxter -> crc_zeller | 3 | 0.774 -> 0.679 | 0.846 -> 0.791 | -0.114 .. -0.002 | 0.826 | 0.512 |
| crc_baxter -> crc_zackular | 3 | 0.774 -> 0.679 | 0.796 -> 0.739 | -0.144 .. +0.024 | 0.777 | 0.567 |
| crc_baxter -> crc_xiang | 3 | 0.774 -> 0.679 | 0.738 -> 0.685 | -0.133 .. +0.012 | 0.705 | 0.654 |
| crc_baxter -> crc_zhao | 3 | 0.774 -> 0.679 | 0.702 -> 0.726 | -0.043 .. +0.091 | 0.676 | 0.464 |
| mdd_tdbrain -> mumtaz | 1 | 0.871 -> 0.984 | 0.250 -> 0.280 | -0.123 .. +0.190 | 0.283 | 0.719 |
| mdd_tdbrain -> mumtaz | 2 | 0.792 -> 0.933 | 0.268 -> 0.587 | +0.120 .. +0.518 | 0.474 | 0.406 |
| mdd_tdbrain -> mumtaz | 3 | 0.887 -> 0.944 | 0.290 -> 0.615 | +0.154 .. +0.502 | 0.315 | 0.355 |
| sz_aseeg -> msu | 1 | 0.699 -> 0.744 | 0.696 -> 0.680 | -0.108 .. +0.077 | 0.625 | 0.659 |
| sz_aseeg -> repod | 1 | 0.699 -> 0.744 | 0.490 -> 0.531 | -0.182 .. +0.262 | 0.526 | 0.439 |
| sz_aseeg -> msu | 2 | 0.583 -> 0.795 | 0.681 -> 0.520 | -0.276 .. -0.043 | 0.684 | 0.700 |
| sz_aseeg -> repod | 2 | 0.583 -> 0.795 | 0.500 -> 0.429 | -0.231 .. +0.109 | 0.510 | 0.548 |
| sz_aseeg -> msu | 3 | 0.750 -> 0.859 | 0.710 -> 0.414 | -0.438 .. -0.147 | 0.681 | 0.705 |
| sz_aseeg -> repod | 3 | 0.750 -> 0.859 | 0.505 -> 0.699 | -0.052 .. +0.448 | 0.469 | 0.510 |
| abnormal_tuab -> tuab_official_eval | 1 | 0.840 -> 0.931 | 0.867 -> 0.930 | +0.034 .. +0.096 | 0.891 | 0.900 |
| abnormal_tuab -> tuab_official_eval | 2 | 0.829 -> 0.899 | 0.865 -> 0.933 | +0.035 .. +0.104 | 0.892 | 0.899 |
| abnormal_tuab -> tuab_official_eval | 3 | 0.872 -> 0.926 | 0.869 -> 0.921 | +0.020 .. +0.084 | 0.918 | 0.897 |

Mean over splits:

| task -> external | internal gain | external gain | external AUROC seed -> loop |
|---|---|---|---|
| abnormal_tuab -> tuab_official_eval | +0.071 | +0.061 | 0.867 -> 0.928 |
| crc_baxter -> crc_xiang | -0.055 | -0.053 | 0.750 -> 0.697 |
| crc_baxter -> crc_zackular | -0.055 | -0.024 | 0.797 -> 0.773 |
| crc_baxter -> crc_zeller | -0.055 | -0.027 | 0.826 -> 0.799 |
| crc_baxter -> crc_zhao | -0.055 | -0.008 | 0.703 -> 0.694 |
| mdd_tdbrain -> mumtaz | +0.104 | +0.225 | 0.269 -> 0.494 |
| sz_aseeg -> msu | +0.122 | -0.158 | 0.696 -> 0.538 |
| sz_aseeg -> repod | +0.122 | +0.054 | 0.498 -> 0.553 |

Notes: the MDD seed is *inverted* on Mumtaz (AUROC < 0.5), i.e. the TDBRAIN case/control
signal is dataset-specific (recording protocol, population); MSU is adolescents with 1-min
16-channel recordings (age/protocol shift from adult ASEEG); RepOD is a small (14+14) adult set;
the TUAB official eval split is the same hospital and protocol (a near-in-distribution control).
