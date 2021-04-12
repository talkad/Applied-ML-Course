import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 2a - performing pre-processing
df = pd.read_excel("dataset.xlsx", engine="openpyxl")

df = df[["SARS-Cov-2 exam result", "Hematocrit", "Hemoglobin", "Platelets", "Red blood Cells", "Lymphocytes",
         "Mean corpuscular hemoglobin (MCH)", "Mean corpuscular hemoglobin concentration (MCHC)",
         "Leukocytes", "Basophils", "Eosinophils", "Lactic Dehydrogenase", "Mean corpuscular volume (MCV)",
         "Red blood cell distribution width (RDW)", "Monocytes", "Mean platelet volume ", "Neutrophils",
         "Proteina C reativa mg/dL", "Creatinine", "Urea", "Potassium", "Sodium", "Aspartate transaminase",
         "Alanine transaminase"]]

df.dropna(thresh=23*0.05, inplace=True)  # filtering rows with more than one empty cell (~95% threshold)

covid_results = df["SARS-Cov-2 exam result"]
df.drop(columns=["SARS-Cov-2 exam result"], inplace=True)

imputer = IterativeImputer(max_iter=250)
imputer.fit(df)
imputed_df = imputer.transform(df)
imputed_df = pd.DataFrame(imputed_df, columns=df.columns)

binary_labels = covid_results.apply(lambda res: 0 if res == 'negative' else 1)

# filtered_df.to_excel("asd.xlsx", engine="openpyxl")


# 2b


