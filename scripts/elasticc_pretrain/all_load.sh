#python train_ray.py --exp_setting def_Syn_exp1_sec1_var --arch_gen arch_gen1 --arch_spec arch_spec1 --search_setting search_enc --selec_col 10 14 18 22
#python train_ray_final.py --exp_setting E_load --arch_gen E --arch_spec E --search_setting E_lc --eval_again ClassifierModelE --load_pretrain E_E_E_E_lc

# lc
python train_ray_final.py --general E --general_e load --searching E_lc --searching_e E --eval_again ClassifierModelE --load_pretrain results/E_lc --name_exp E_lc_load
# tab
python train_ray_final.py --general E --general_e load --searching E_tab --searching_e E --eval_again ClassifierModelE --load_pretrain results/E_tab --name_exp E_tab_load
