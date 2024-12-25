import pandas as pd
import pickle
from fastapi import FastAPI, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, Float, String, Date, Table, MetaData
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler
from category_encoders import BinaryEncoder

# --------- Configuration ---------
DATABASE_URL = "postgresql+asyncpg://postgres:m123456789m@localhost/test6"
XGBOOST_MODEL_PATH = "xgb_best_model.pkl"
CLEANED_CSV_PATH = "cleaned_data.csv"
OUTPUT_DIRECTORY = "data"
app = FastAPI(title="Batch Preprocessing and Prediction API")

# --------- Database Setup ---------
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

metadata = MetaData()

# Table de prédiction
table_predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("store", Integer),
    Column("date", Date),
    Column("predicted_sales", Float),
)


# --------- Utility Functions ---------
def get_season(month):
    """Get season according to the month."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


def preprocess_data(df: pd.DataFrame):
    """Preprocess input data."""
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["Season"] = df["Month"].apply(get_season)

    # Ajouter des colonnes de lag
    df["Lag_1"] = df.groupby("Store")["Weekly_Sales"].shift(1)
    df["Lag_2"] = df.groupby("Store")["Weekly_Sales"].shift(2)
    df["Lag_3"] = df.groupby("Store")["Weekly_Sales"].shift(3)
    df["Lag_4"] = df.groupby("Store")["Weekly_Sales"].shift(4)

    df.fillna(0, inplace=True)

    # Suppression des outliers
    num_features = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Weekly_Sales"]
    for feature in num_features:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[feature] >= lower) & (df[feature] <= upper)]

    # Standardisation des variables numériques
    num_vars = [
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Weekly_Sales",
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "Lag_4",
    ]
    sc = StandardScaler()
    df[num_vars] = sc.fit_transform(df[num_vars])
    df["Original_Store"] = df["Store"]
    df["Original_Date"] = df["Date"]    
    # Encodage des variables catégorielles
    encoder = BinaryEncoder(cols=["Store", "Season"])
    df = encoder.fit_transform(df)

    drop_columns = ["Weekly_Sales", "Date", "Year"]
    df = df.drop(columns=drop_columns)

    return df


# --------- Load Model ---------
with open(XGBOOST_MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# --------- Endpoints ---------
@app.post("/preprocess/", tags=["Preprocessing"])
async def preprocess_endpoint(file: UploadFile = File(...)):
    """Endpoint to upload a CSV file, preprocess it, and append cleaned data to cleaned_data.csv."""
    try:
        # Charger les données à partir du fichier téléchargé
        df = pd.read_csv(file.file)

        # Nettoyer les données
        processed_df = preprocess_data(df)

        # Charger les données nettoyées existantes si le fichier existe
        if os.path.exists(CLEANED_CSV_PATH):
            existing_data = pd.read_csv(CLEANED_CSV_PATH)
            # Ajouter les nouvelles données
            combined_data = pd.concat([existing_data, processed_df], ignore_index=True)
        else:
            combined_data = processed_df

        # Sauvegarder toutes les données dans le fichier cleaned_data.csv
        combined_data.to_csv(CLEANED_CSV_PATH, index=False)

        return {
            "message": "Data successfully preprocessed and appended to cleaned_data.csv.",
            "file": CLEANED_CSV_PATH,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing data: {str(e)}")



@app.post("/predict/", tags=["Prediction"])
async def predict_endpoint():
    """Endpoint to predict sales and save results into the database."""
    try:
        # Vérifiez si le fichier de données nettoyées existe
        if not os.path.exists(CLEANED_CSV_PATH):
            raise HTTPException(
                status_code=400,
                detail="Cleaned data file not found. Run preprocessing first.",
            )

        # Charger les données nettoyées
        df = pd.read_csv(CLEANED_CSV_PATH)

        # Vérifiez si toutes les colonnes nécessaires sont présentes
        required_columns = [
            "Store_0",
            "Store_1",
            "Store_2",
            "Store_3",
            "Store_4",
            "Store_5",
            "Holiday_Flag",
            "Temperature",
            "Fuel_Price",
            "CPI",
            "Unemployment",
            "Month",
            "Season_0",
            "Season_1",
            "Season_2",
            "Lag_1",
            "Lag_2",
            "Lag_3",
            "Lag_4",
        ]
        for col in required_columns:
            if col not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required column '{col}' in cleaned data. Check preprocessing.",
                )

        # Préparer les features pour la prédiction
        X = df[required_columns]

        # Faire la prédiction
        predictions = model.predict(X)
        df["Predicted_Sales"] = predictions
        save_data = df[["Original_Store", "Original_Date", "Predicted_Sales"]]

        # Sauvegarder les prédictions dans un fichier CSV
        predictions_csv_path = "predictions.csv"
        save_data.to_csv(predictions_csv_path, index=False)

        return {
            "message": "Predictions saved successfully to CSV file.",
            "file": predictions_csv_path,
        }

        # Ajouter les prédictions aux données
        # df["Predicted_Sales"] = predictions
        # save_data = df[["Original_Store", "Original_Date", "Predicted_Sales"]]

        # # Sauvegarder les prédictions dans la base de données
        # async with async_session() as session:
        #     async with session.begin():
        #         for _, row in save_data.iterrows():
        #             await session.execute(
        #                 table_predictions.insert().values(
        #                     Store=row["Original_Store"],
        #                     Date=datetime.strptime(row["Original_Date"], "%Y-%m-%d"),
        #                     predicted_sales=float(row["Predicted_Sales"]),
        #                 )
        #             )

        # return {"message": "Predictions saved successfully to database."}

    except Exception as e:
        # Gestion des erreurs avec un message clair
        raise HTTPException(
            status_code=400, detail=f"Error during prediction: {str(e)}"
        )

@app.get("/results/", tags=["Results"])
async def fetch_results():
    """Endpoint to fetch predictions from the predictions.csv file."""
    try:
        # Vérifiez si le fichier existe
        if not os.path.exists("predictions.csv"):
            raise HTTPException(
                status_code=404,
                detail="Predictions file not found. Run the prediction endpoint first.",
            )
        
        # Lire le fichier CSV
        predictions_df = pd.read_csv("predictions.csv")
        
        # Convertir le DataFrame en liste de dictionnaires pour JSON
        predictions_list = predictions_df.to_dict(orient="records")
        
        return {
            "message": "Predictions fetched successfully.",
            "predictions": predictions_list,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching predictions: {str(e)}")

# @app.get("/results/", tags=["Results"])
# async def fetch_results():
#     """Endpoint to fetch predictions from the database."""
#     try:
#         async with async_session() as session:
#             result = await session.execute(table_predictions.select())
#             predictions = result.fetchall()
#         return predictions
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))


# --------- Run Server ---------
# Run with: uvicorn main:app --reload
@app.post("/split_csv/", tags=["CSV Processing"])
async def split_csv(file: UploadFile = File(...)):
    """
    Endpoint to upload a CSV file, split its rows evenly across 10 CSV files with all stores,
    and save them in the specified directory.
    """
    try:
        # Charger les données du fichier CSV
        df = pd.read_csv(file.file)

        # Vérification si le fichier a suffisamment de lignes
        if df.shape[0] < 1000:
            raise HTTPException(
                status_code=400, detail="The uploaded CSV file must have at least 1000 rows."
            )

        # Mélanger les lignes pour répartir les stores de manière aléatoire
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Initialiser les fichiers de sortie
        chunks = [pd.DataFrame() for _ in range(10)]

        # Répartir les lignes entre les fichiers
        for idx, row in df.iterrows():
            target_chunk = idx % 10  # Assigner chaque ligne à l'un des 10 fichiers de manière cyclique
            chunks[target_chunk] = pd.concat([chunks[target_chunk], pd.DataFrame([row])])

        # Sauvegarder les fichiers
        file_paths = []
        for i, chunk in enumerate(chunks):
            output_file_path = os.path.join(OUTPUT_DIRECTORY, f"fichier{i + 1}.csv")
            chunk.to_csv(output_file_path, index=False)
            file_paths.append(output_file_path)

        return {
            "message": "The file has been successfully split into 10 CSV files.",
            "files": file_paths,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing the file: {str(e)}")
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
