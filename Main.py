# run_stepwise.py
from Base import fit_stepwise_models

results = fit_stepwise_models(
    xlsx_path=r"C:\Users\devli\OneDrive - Imperial College London\MSci - Devlin (Personal)\Data\FP_db_all.xlsx",   # <-- put your Excel path here
    sheet_db="DB",
    sheet_psd="PSD",
    target_moisture="Mc_%",
    target_porosity="Cake_por",
    test_size=0.2,
    random_state=42,
    verbose=True
)

print("\nSelected (moisture):", results["res_moisture"]["selected_features"])
print("Selected (porosity):", results["res_porosity"]["selected_features"])
