# LC
python train.py --general ELASTICC --general_e train --searching lc --searching_e spec --selec_col 10 14 18 22 --name_exp lc
# TAB
python train.py --general ELASTICC --general_e train --searching tab --searching_e spec --selec_col 10 14 18 22 --name_exp tab
# LC + TAB
python train.py --general ELASTICC --general_e train --searching lc_tab --searching_e spec --selec_col 10 14 18 22 --name_exp lc_tab
# (LC + TAB) ablation
python train.py --general ELASTICC --general_e train --searching ablation --searching_e spec --selec_col 10 14 18 22 --name_exp ablation