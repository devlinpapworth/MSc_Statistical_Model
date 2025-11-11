# run_stepwise.py
from base_code.Base import fit_stepwise_models
from base_code.Export_EQ import print_formulas
from Plotting.plot_stepwise_results import plots_main

def main():
    results = fit_stepwise_models(
        xlsx_path=r"C:\Users\devli\OneDrive - Imperial College London\MSci - Devlin (Personal)\Data\FP_db_all.xlsx",
        sheet_db="DB",
        sheet_psd="PSD",
        target_moisture="Mc_%",
        target_porosity="Cake_por",
        test_size=0.2, #20% of data used for testing model
        random_state=42, # seed for repeadtability
        verbose=True
    )

    print("\nSelected (moisture):", results["res_moisture"]["selected_features"])
    print("Selected (porosity):", results["res_porosity"]["selected_features"])

    # === Export raw-input formulas ===
    print_formulas(results)

    # this reads stepwise_predictions.csv and shows the plots
    plots_main(csv_path="stepwise_predictions.csv", save_path="model_fit")

if __name__ == "__main__":
    main()
