from src.Ingestion import create_spark_session, load_data
from Health_visualisation.src.Processing import Data_profile, clean_data
from src.Analysis import (
    descriptive_analysis,
    correlation_analysis,
    hypothesis_testing,
    risk_stratification,
    export_to_powerbi
)


def run_pipeline():
    
    print("=" * 60)
    print("Healthcare Data ETL Pipeline")
    print("=" * 60)
    
    # Initialize Spark
    print("\n[1/6] Initializing Spark session...")
    spark = create_spark_session()
    
    # Load Data
    print("\n[2/6] Loading data from S3...")
    raw_df = load_data(spark)
    
    # Profile Raw Data
    print("\n[3/6] Profiling raw data...")
    Data_profile(raw_df, "Raw Heart Disease Data")
    
    # Clean Data
    print("\n[4/6] Cleaning data...")
    cleaned_df = clean_data(raw_df)
    
    # Run Analysis
    print("\n[5/6] Running statistical analysis...")
    descriptive_analysis(cleaned_df, spark)
    correlation_analysis(cleaned_df, spark)
    hypothesis_testing(cleaned_df, spark)
    risk_stratification(cleaned_df)
    
    # Export to Power BI
    print("\n[6/6] Exporting to Power BI...")
    export_to_powerbi(cleaned_df, spark)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    
    return cleaned_df


if __name__ == "__main__":
    run_pipeline()