def Data_profile(df, name="DataFrame"):
    """Generate comprehensive data profile"""
    print(f"\n=== PROFILE: {name} ===")
    
    total_rows = df.count()
    total_cols = len(df.columns)
    print(f"Rows: {total_rows}, Columns: {total_cols}")
    
    # Null analysis - use spark_sum instead of sum
    print("\n--- Null/Missing Values ---")
    null_counts = df.select([
        spark_sum(
            when(col(c).isNull() | (col(c) == "") | (col(c) == "NA"), 1).otherwise(0)
        ).alias(c) for c in df.columns
    ])
    null_counts.show(truncate=False)
    
    # Convert to percentage - use spark_round instead of round
    null_pct = df.select([
        spark_round(
            spark_sum(when(col(c).isNull() | (col(c) == ""), 1).otherwise(0)) / total_rows * 100, 2
        ).alias(c) for c in df.columns
    ])
    print("--- Null Percentage ---")
    null_pct.show(truncate=False)
    
    # Numeric column stats
    print("\n--- Numeric Statistics ---")
    numeric_cols = [f.name for f in df.schema.fields 
                    if str(f.dataType) in ['IntegerType()', 'DoubleType()', 'LongType()']]
    if numeric_cols:
        df.select(numeric_cols).describe().show()
    
    # Categorical value counts
    print("\n--- Categorical Distributions ---")
    categorical_cols = ['sex', 'dataset', 'cp', 'restecg', 'slope', 'thal']
    for col_name in categorical_cols:
        if col_name in df.columns:
            print(f"\n{col_name}:")
            df.groupBy(col_name).count().orderBy('count', ascending=False).show(10)
    
    return null_counts

def clean_data(df):
    """Data cleaning and standardization"""
    
    # 1. Handle boolean columns (fbs, exang)
    df_cleaned = df.withColumn("fbs", 
        when(upper(col("fbs")) == "TRUE", True)
        .when(upper(col("fbs")) == "FALSE", False)
        .otherwise(None).cast(BooleanType())
    ).withColumn("exang",
        when(upper(col("exang")) == "TRUE", True)
        .when(upper(col("exang")) == "FALSE", False)
        .otherwise(None).cast(BooleanType())
    )
    
    # 2. Columns where 0 means missing
    zero_is_null_cols = ['trestbps', 'chol', 'thalch']
    for col_name in zero_is_null_cols:
        df_cleaned = df_cleaned.withColumn(col_name, 
            when((col(col_name) == "") | (col(col_name) == 0), None)
            .otherwise(col(col_name)).cast(DoubleType())
        )
    
    # 3. Columns where 0 is valid
    zero_is_valid_cols = ['age', 'oldpeak', 'ca', 'num']
    for col_name in zero_is_valid_cols:
        df_cleaned = df_cleaned.withColumn(col_name, 
            when(col(col_name) == "", None)
            .otherwise(col(col_name)).cast(DoubleType())
        )
    
    # 4. Standardize categorical values
    df_cleaned = df_cleaned \
        .withColumn("sex", trim(col("sex"))) \
        .withColumn("cp", trim(col("cp"))) \
        .withColumn("restecg", trim(col("restecg"))) \
        .withColumn("slope", trim(col("slope"))) \
        .withColumn("thal", trim(col("thal"))) \
        .withColumn("dataset", trim(col("dataset")))
    
    # 5. Calculate means for imputation (use spark_round)
    means = df_cleaned.select(
        *[spark_round(avg(col(c)), 1).alias(c) for c in ['trestbps', 'chol', 'thalch', 'oldpeak']]
    ).collect()[0]
    
    print(f"Imputation values: trestbps={means['trestbps']}, chol={means['chol']}, thalch={means['thalch']}, oldpeak={means['oldpeak']}")
    
    # 6. Impute missing numeric values
    df_cleaned = df_cleaned \
        .withColumn("trestbps", coalesce(col("trestbps"), lit(means['trestbps']))) \
        .withColumn("chol", coalesce(col("chol"), lit(means['chol']))) \
        .withColumn("thalch", coalesce(col("thalch"), lit(means['thalch']))) \
        .withColumn("oldpeak", coalesce(col("oldpeak"), lit(means['oldpeak'])))
    
    # 7. Fill categorical nulls with 'unknown'
    df_cleaned = df_cleaned \
        .withColumn("thal", when(col("thal").isNull() | (col("thal") == ""), "unknown").otherwise(col("thal"))) \
        .withColumn("slope", when(col("slope").isNull() | (col("slope") == ""), "unknown").otherwise(col("slope"))) \
        .withColumn("ca", when(col("ca").isNull(), -1).otherwise(col("ca")))  # Use -1 for missing ca
    
    return df_cleaned