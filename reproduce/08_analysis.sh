#!/bin/bash
# Step 8: rebuild the tables and figures from the saved probabilities (CPU only). Results go to $DSRE_ROOT/results.
source "$(dirname "$0")/lib.sh"
cd $CODE/analysis
python3 table2.py                       # Tables 2-3
python3 table2_nyt10m.py                # NYT10m columns of Table 2
python3 tables_cpu.py                   # Table 7, Table 8, data of Figure 6
python3 cmax_sensitivity.py 0.1,0.2,0.25,0.3,0.4   # Table 10
python3 noise_types.py                  # Table 5
python3 error_analysis.py               # Section 4.2.1
python3 ablation_summary.py             # Tables 6 and 9
python3 fig_data.py --ds nyt10; python3 fig_data.py --ds gids; python3 plot_figs.py; python3 plot_fig6.py   # Figures 5-7
