# Exploration and Statistical Analysis

import os
os.environ.pop('CONTAINER_ID', None)

import pandas as pd
import numpy as np
from scipy import stats

from pyspark.sql.functions import (
    col, when, count, avg, stddev, variance,
    min as spark_min, max as spark_max, corr,
    sum as spark_sum, round as spark_round, lit
)
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# Descriptive Statistics
def descriptive_analysis(df, spark):
    """Generate descriptive statistics as DataFrames for Power BI"""
    
    print("=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    
    numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    
    # Base Stats
    stats_list = []
    for c in numeric_cols:
        stats_row = df.select(
            lit(c).alias("variable"),
            spark_round(avg(col(c)), 2).alias("mean"),
            spark_round(stddev(col(c)), 2).alias("std_dev"),
            spark_round(variance(col(c)), 2).alias("variance"),
            spark_round(spark_min(col(c)), 2).alias("min"),
            spark_round(spark_max(col(c)), 2).alias("max"),
        )
        stats_list.append(stats_row)
    
    descriptive_stats_df = stats_list[0]
    for stat_df in stats_list[1:]:
        descriptive_stats_df = descriptive_stats_df.union(stat_df)
    
    print("\n--- Descriptive Statistics Table ---")
    descriptive_stats_df.show(truncate=False)
    
    # Categorical Frequencies
    categorical_cols = ['sex', 'cp', 'dataset', 'restecg', 'slope', 'thal']
    total_count = df.count()
    
    freq_dfs = []
    for cat_col in categorical_cols:
        freq_df = df.groupBy(cat_col).agg(
            count("*").alias("count"),
            spark_round(count("*") / total_count * 100, 2).alias("percentage")
        ).withColumn("variable", lit(cat_col)) \
         .withColumnRenamed(cat_col, "category") \
         .select("variable", "category", "count", "percentage")
        freq_dfs.append(freq_df)
    
    categorical_freq_df = freq_dfs[0]
    for f_df in freq_dfs[1:]:
        categorical_freq_df = categorical_freq_df.union(f_df)
    
    print("\n--- Categorical Frequencies Table ---")
    categorical_freq_df.show(50, truncate=False)
    
    return descriptive_stats_df, categorical_freq_df

# Correlation Matrix
def correlation_analysis(df, spark):
    """Generate correlation matrix as DataFrame for Power BI"""
    
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Add binary target
    df_with_target = df.withColumn("has_heart_disease", 
        when(col("num") > 0, 1.0).otherwise(0.0)
    )
    
    numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'has_heart_disease']
    
    # correlation matrix
    corr_rows = []
    for col1 in numeric_cols:
        row_data = {"variable": col1}
        for col2 in numeric_cols:
            corr_val = df_with_target.stat.corr(col1, col2)
            row_data[col2] = round(corr_val, 4)
        corr_rows.append(row_data)
    
    corr_df = spark.createDataFrame(corr_rows)
    
    print("\n--- Correlation Matrix ---")
    corr_df.show(truncate=False)
    
    # Correlation with target
    target_corr_rows = []
    for c in numeric_cols[:-1]:
        corr_val = df_with_target.stat.corr(c, "has_heart_disease")
        abs_corr = abs(corr_val)
        
        if abs_corr >= 0.5:
            strength = "Strong"
        elif abs_corr >= 0.3:
            strength = "Moderate"
        elif abs_corr >= 0.1:
            strength = "Weak"
        else:
            strength = "Very Weak"
        
        direction = "Positive" if corr_val > 0 else "Negative"
        
        target_corr_rows.append({
            "variable": c,
            "correlation": round(corr_val, 4),
            "abs_correlation": round(abs_corr, 4),
            "strength": strength,
            "direction": direction
        })
    
    target_corr_df = spark.createDataFrame(target_corr_rows)
    
    print("\n--- Correlation with Heart Disease ---")
    target_corr_df.show(truncate=False)
    
    return corr_df, target_corr_df

# Hypothesis Testing
def hypothesis_testing(df, spark):
    """Generate hypothesis testing results as DataFrames for Power BI"""
    
    print("\n" + "=" * 60)
    print("HYPOTHESIS TESTING")
    print("=" * 60)
    
    df_with_target = df.withColumn("has_heart_disease", 
        when(col("num") > 0, 1).otherwise(0)
    )
    pdf = df_with_target.toPandas()
    
    alpha = 0.05
    
    # T-Tests
    print("\n--- T-Test Results ---")
    
    continuous_vars = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    disease_group = pdf[pdf['has_heart_disease'] == 1]
    no_disease_group = pdf[pdf['has_heart_disease'] == 0]
    
    ttest_results = []
    for var in continuous_vars:
        group1 = no_disease_group[var].dropna()
        group2 = disease_group[var].dropna()
        
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        ttest_results.append({
            "variable": var,
            "test_type": "Independent T-Test",
            "mean_no_disease": round(float(group1.mean()), 2),
            "mean_disease": round(float(group2.mean()), 2),
            "mean_difference": round(float(group2.mean() - group1.mean()), 2),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 6),
            "significant": "Yes" if p_value < alpha else "No",
            "significance_level": float(alpha)
        })
    
    ttest_df = spark.createDataFrame(ttest_results)
    ttest_df.show(truncate=False)

    # Chi-Square Test
    print("\n--- Chi-Square Test Results ---")
    
    categorical_vars = ['sex', 'cp', 'restecg', 'slope', 'thal']
    
    chi2_results = []
    for var in categorical_vars:
        contingency = pd.crosstab(pdf[var], pdf['has_heart_disease'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        # Cramér's V
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        if cramers_v >= 0.5:
            effect = "Large"
        elif cramers_v >= 0.3:
            effect = "Medium"
        elif cramers_v >= 0.1:
            effect = "Small"
        else:
            effect = "Negligible"
        
        chi2_results.append({
            "variable": var,
            "test_type": "Chi-Square",
            "chi2_statistic": round(float(chi2), 4),
            "degrees_of_freedom": int(dof),
            "p_value": round(float(p_value), 6),
            "cramers_v": round(float(cramers_v), 4),
            "effect_size": effect,
            "significant": "Yes" if p_value < alpha else "No"
        })
    
    chi2_df = spark.createDataFrame(chi2_results)
    chi2_df.show(truncate=False)

    # Group comparison
    print("\n--- Group Comparison Summary ---")
    
    group_summary = df_with_target.groupBy("has_heart_disease").agg(
        count("*").alias("count"),
        spark_round(avg("age"), 2).alias("avg_age"),
        spark_round(avg("trestbps"), 2).alias("avg_bp"),
        spark_round(avg("chol"), 2).alias("avg_cholesterol"),
        spark_round(avg("thalch"), 2).alias("avg_max_hr"),
        spark_round(avg("oldpeak"), 2).alias("avg_st_depression")
    ).withColumn("group", 
        when(col("has_heart_disease") == 0, "No Disease").otherwise("Heart Disease")
    )
    
    group_summary.show(truncate=False)
    
    return ttest_df, chi2_df, group_summary

# Risk Stratifiation
def risk_stratification(df):
    """Generate risk stratification tables for Power BI"""
    
    print("\n" + "=" * 60)
    print("RISK STRATIFICATION")
    print("=" * 60)
    
    df_risk = df.withColumn("has_heart_disease", 
        when(col("num") > 0, 1).otherwise(0)
    ).withColumn("age_group",
        when(col("age") < 40, "Under 40")
        .when(col("age") < 50, "40-49")
        .when(col("age") < 60, "50-59")
        .when(col("age") < 70, "60-69")
        .otherwise("70+")
    ).withColumn("risk_score",
        (when(col("age") >= 55, 1).otherwise(0) +
         when(col("sex") == "Male", 1).otherwise(0) +
         when(col("cp") == "asymptomatic", 2).otherwise(0) +
         when(col("trestbps") >= 140, 1).otherwise(0) +
         when(col("chol") >= 240, 1).otherwise(0) +
         when(col("fbs") == True, 1).otherwise(0) +
         when(col("exang") == True, 1).otherwise(0) +
         when(col("oldpeak") >= 2, 1).otherwise(0))
    ).withColumn("risk_level",
        when(col("risk_score") <= 2, "Low")
        .when(col("risk_score") <= 4, "Medium")
        .otherwise("High")
    ).withColumn("severity",
        when(col("num") == 0, "None")
        .when(col("num") == 1, "Mild")
        .when(col("num") == 2, "Moderate")
        .otherwise("Severe")
    )
    
    # Risk by Age Group
    print("\n--- Risk by Age Group ---")
    age_risk = df_risk.groupBy("age_group").agg(
        count("*").alias("total_count"),
        spark_sum("has_heart_disease").alias("disease_count"),
        spark_round(avg("has_heart_disease") * 100, 2).alias("disease_rate_pct"),
        spark_round(avg("risk_score"), 2).alias("avg_risk_score")
    ).orderBy("age_group")
    age_risk.show()
    
    # Risk by Sex
    print("\n--- Risk by Sex ---")
    sex_risk = df_risk.groupBy("sex").agg(
        count("*").alias("total_count"),
        spark_sum("has_heart_disease").alias("disease_count"),
        spark_round(avg("has_heart_disease") * 100, 2).alias("disease_rate_pct"),
        spark_round(avg("risk_score"), 2).alias("avg_risk_score")
    )
    sex_risk.show()
    
    # Risk Level Distribution
    print("\n--- Risk Level Distribution ---")
    risk_level_dist = df_risk.groupBy("risk_level").agg(
        count("*").alias("total_count"),
        spark_sum("has_heart_disease").alias("disease_count"),
        spark_round(avg("has_heart_disease") * 100, 2).alias("disease_rate_pct")
    ).orderBy(
        when(col("risk_level") == "Low", 1)
        .when(col("risk_level") == "Medium", 2)
        .otherwise(3)
    )
    risk_level_dist.show()
    
    # Severity Distribution
    print("\n--- Severity Distribution ---")
    severity_dist = df_risk.groupBy("severity").agg(
        count("*").alias("count"),
        spark_round(avg("age"), 2).alias("avg_age"),
        spark_round(avg("trestbps"), 2).alias("avg_bp"),
        spark_round(avg("chol"), 2).alias("avg_cholesterol")
    )
    severity_dist.show()
    
    # Cross-tabulation: Age Group x Sex
    print("\n--- Disease Rate: Age Group x Sex ---")
    cross_tab = df_risk.groupBy("age_group", "sex").agg(
        count("*").alias("count"),
        spark_round(avg("has_heart_disease") * 100, 2).alias("disease_rate_pct")
    ).orderBy("age_group", "sex")
    cross_tab.show()
    
    return df_risk, age_risk, sex_risk, risk_level_dist, severity_dist, cross_tab

# Exports the analysis tables PowerBI
def export_to_powerbi(cleaned_df, spark, output_path="s3a://pritham-heartdata/powerbi/"):
    """Export all analysis results to CSV for Power BI"""
    
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS FOR POWER BI")
    print("=" * 60)
    
    descriptive_stats_df, categorical_freq_df = descriptive_analysis(cleaned_df, spark)
    corr_matrix_df, target_corr_df = correlation_analysis(cleaned_df, spark)
    ttest_df, chi2_df, group_summary = hypothesis_testing(cleaned_df, spark)
    transformed_df, age_risk, sex_risk, risk_level_dist, severity_dist, cross_tab = risk_stratification(cleaned_df)
    
    # Exports all tables
    exports = {
        "01_descriptive_stats": descriptive_stats_df,
        "02_categorical_frequencies": categorical_freq_df,
        "03_correlation_matrix": corr_matrix_df,
        "04_target_correlations": target_corr_df,
        "05_ttest_results": ttest_df,
        "06_chi_square_results": chi2_df,
        "07_group_summary": group_summary,
        "08_risk_by_age_group": age_risk,
        "09_risk_by_sex": sex_risk,
        "10_risk_level_distribution": risk_level_dist,
        "11_severity_distribution": severity_dist,
        "12_age_sex_crosstab": cross_tab,
        "13_full_transformed_data": transformed_df
    }
    
    for name, df in exports.items():
        path = f"{output_path}{name}"
        df.coalesce(1).write.mode("overwrite").option("header", True).csv(path)
        print(f"✓ Exported: {name}")
    
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE!")
    print("=" * 60)
    print(f"\nFiles saved to: {output_path}")
    print("\nConnect Power BI to S3 or download CSVs to import.")
    
    return exports