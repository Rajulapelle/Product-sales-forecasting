### Core Problem

Retail store managers and supply chain executives face a constant "optimization dilemma" concerning sales forecasting:

  Under-stocking: Leads to stockouts, immediate lost revenue, and long-term damage to customer loyalty.

  Over-stocking: Results in inflated inventory costs, potential waste (especially for perishable goods), and profit erosion due to clearance discounts. Sales fluctuations are complex, influenced by non-linear interactions of time (weekends, holidays), location, store attributes, and marketing interventions (discounts).

**Project Objective**

The primary objective of this project is to develop a robust Machine Learning Regression Model. This model will be capable of predicting future sales for specific retail outlets by analyzing historical transaction data to learn patterns and relationships between sales figures and various independent variables.

**Expected Outcomes & Business Value**

The successful deployment of this forecasting model is expected to provide significant business value by addressing the following needs:

  Inventory Optimization: Ensuring the right amount of stock is available at the right store at the right time.

  Revenue Projection: Assisting finance teams in accurately estimating future cash flows.

  Resource Allocation: Helping management make informed decisions on staffing levels and logistics support during predicted peak periods.

  Marketing Strategy: Identifying which store types and regions respond best to promotions, thereby enabling more targeted marketing campaigns.

  ### Target Metric: Sales

The main dependent variable or target metric for this project is **'Sales'**. This aligns directly with the project objective, which is to develop a robust Machine Learning Regression Model capable of predicting future sales for specific retail outlets.

## Summary of EDA Steps


1.  **Univariate Analysis (Numerical Features):** Histograms and box plots were generated for 'Sales' and '#Order'. Key observations included that both 'Sales' and '#Order' distributions were right-skewed, indicating a few high-value transactions/orders and many lower ones. The median sales was approximately $32,000-$33,000, and the median number of orders was 50-60, with significant outliers on the higher end for both.
2.  **Univariate Analysis (Categorical Features):** Count plots were used to visualize the distributions of 'Store_Type', 'Location_Type', 'Region_Code', 'Holiday', and 'Discount'. Observations showed varying counts across 'Location_Type' and 'Region_Code'. 'Holiday' was imbalanced (more non-holidays), while 'Discount' was relatively balanced.
3.  **Time Series Analysis of Sales:** The 'Date' column was converted to datetime objects, and daily total sales were calculated and plotted using a line plot. This revealed fluctuations and potential seasonality in sales over time.
4.  **Bivariate Analysis (Sales vs. Categorical Factors):** Box plots were employed to examine 'Sales' against 'Discount', 'Holiday', 'Store_Type', 'Location_Type', and 'Region_Code'. It was observed that sales with discounts and on holidays generally had higher medians and wider spreads. Different categories within 'Store_Type', 'Location_Type', and 'Region_Code' showed varying sales distributions and median values.
5.  **Numerical Correlation:** A heatmap was generated to visualize the Pearson correlation between 'Sales' and '#Order'. A very strong positive correlation (approx. 0.94) was found, indicating that as the number of orders increased, sales also increased proportionally.
6.  **Outlier Detection:** Box plots and the Interquartile Range (IQR) method were used to identify outliers in 'Sales' and '#Order'. It was found that 5843 outliers existed in 'Sales' (outside -1798.50 to 84133.50) and 7089 outliers in '#Order' (outside -3.00 to 133.00), suggesting frequent occurrences of unusually high transactions or orders.

### Summary of Hypothesis Testing Steps

1.  **Impact of Discounts on Sales (Independent Samples t-test)**:
    *   **Null Hypothesis (H0)**: There is no significant difference in the mean sales between days with discounts and days without discounts.
    *   **Alternative Hypothesis (Ha)**: There is a significant difference in the mean sales between days with discounts and days without discounts.
    *   **T-statistic**: 145.93
    *   **P-value**: 0.000
    *   **Conclusion**: Since the p-value (0.000) is less than the significance level (0.05), we reject the null hypothesis. There is a significant difference in mean sales between days with discounts (Mean: $49426.50) and days without discounts (Mean: $37403.68).

2.  **Regional Sales Variability (Kruskal-Wallis H-test)**:
    *   **Null Hypothesis (H0)**: There is no significant difference in the median sales across different regions.
    *   **Alternative Hypothesis (Ha)**: There is a significant difference in the median sales across at least one pair of regions.
    *   **H-statistic**: 3968.06
    *   **P-value**: 0.000
    *   **Conclusion**: Since the p-value (0.000) is less than the significance level (0.05), we reject the null hypothesis. There is a significant difference in median sales across different regions (e.g., Region R1: $43125.00, Region R2: $37548.00).

3.  **Effect of Holidays on Sales (Independent Samples t-test)**:
    *   **Null Hypothesis (H0)**: There is no significant difference in the mean sales between holidays and non-holidays.
    *   **Alternative Hypothesis (Ha)**: There is a significant difference in the mean sales between holidays and non-holidays.
    *   **T-statistic**: -66.18
    *   **P-value**: 0.000
    *   **Conclusion**: Since the p-value (0.000) is less than the significance level (0.05), we reject the null hypothesis. There is a significant difference in mean sales between holidays (Mean: $35451.88) and non-holidays (Mean: $43897.29).

4.  **Sales Differences Across Store Types (ANOVA Test)**:
    *   **Null Hypothesis (H0)**: There is no significant difference in the mean sales across different store types.
    *   **Alternative Hypothesis (Ha)**: There is a significant difference in the mean sales across at least one pair of store types.
    *   **F-statistic**: 35123.64
    *   **P-value**: 0.000
    *   **Conclusion**: Since the p-value (0.000) is less than the significance level (0.05), we reject the null hypothesis. There is a significant difference in mean sales across different store types (e.g., S1: $37676.51, S2: $27530.83, S3: $47063.07, S4: $59945.69).

5.  **Correlation between Number of Orders and Sales (Pearson Correlation Test)**:
    *   **Null Hypothesis (H0)**: There is no linear relationship between the number of orders ('#Order') and sales ('Sales'). The Pearson correlation coefficient is equal to 0.
    *   **Alternative Hypothesis (Ha)**: There is a significant linear relationship between the number of orders ('#Order') and sales ('Sales'). The Pearson correlation coefficient is not equal to 0.
    *   **Correlation Coefficient**: 0.94
    *   **P-value**: 0.000
    *   **Conclusion**: Since the p-value (0.000) is less than the significance level (0.05) and the correlation coefficient is 0.94 (very strong and positive), we reject the null hypothesis. There is a statistically significant, very strong, and positive linear relationship between the number of orders and sales, indicating that as the number of orders increases, sales tend to increase proportionally.


    ### Summary of ML Modeling Steps

#### 1. Linear Regression
*   **Training**: A Linear Regression model was trained on the `X_train_split` and `y_train_split` datasets.
*   **Evaluation**: Performance was assessed on the `X_val_split` and `y_val_split` validation sets using metrics such as MAE, MSE, RMSE, R-squared, and MAPE.

#### 2. ARIMA (Autoregressive Integrated Moving Average)
*   **Training**: An ARIMA model with order (5, 1, 0) was fitted to the `train_data` portion of the `daily_sales` time series.
*   **Evaluation**: Predictions were generated for the `validation_data` (which is the time-series validation set), and evaluated using MAE, MSE, RMSE, R-squared, and MAPE.

#### 3. SARIMA (Seasonal Autoregressive Integrated Moving Average)
*   **Training**: A SARIMA model with non-seasonal order (5, 1, 0) and seasonal order (1, 1, 0, 7) was fitted to the `train_data` time series.
*   **Evaluation**: Similar to ARIMA, predictions were made on `validation_data` and evaluated with MAE, MSE, RMSE, R-squared, and MAPE.

#### 4. Prophet
*   **Training**: A Prophet model, configured with yearly and weekly seasonality, was trained on the `prophet_train_data` (a reformatted version of `train_data`).
*   **Evaluation**: Forecasts were generated for the period corresponding to `validation_data` and evaluated using MAE, MSE, RMSE, R-squared, and MAPE.

#### 5. RandomForestRegressor
*   **Training**: A RandomForestRegressor model was trained on the `X_train_split` and `y_train_split` datasets.
*   **Evaluation**: Predictions were made on `X_val_split` and evaluated using MAE, MSE, RMSE, and R-squared.

#### 6. XGBoostRegressor
*   **Training**: An XGBoostRegressor model was trained on the `X_train_split` and `y_train_split` datasets.
*   **Evaluation**: Predictions were made on `X_val_split` and evaluated using MAE, MSE, RMSE, and R-squared.

#### 7. LightGBM Regressor
*   **Training**: A LightGBM Regressor model was trained on the `X_train_split` and `y_train_split` datasets.
*   **Evaluation**: Predictions were made on `X_val_split` and evaluated using MAE, MSE, RMSE, and R-squared.

#### 8. Ensemble Model
*   **Creation**: An ensemble model was created by combining the predictions from the three top-performing individual regression models: RandomForestRegressor, XGBoostRegressor, and LightGBM Regressor. A simple averaging technique was used.
*   **Evaluation**: The ensemble's predictions on `X_val_split` were evaluated against `y_val_split` using MAE, MSE, RMSE, and R-squared.

### Consolidated Insights and Recommendations

Based on the extensive Exploratory Data Analysis (EDA), Hypothesis Testing, and Model Comparison, several key insights and actionable recommendations can be formulated to improve sales forecasting and operational efficiency.

#### Key Insights:

1.  **Sales and Order Dynamics**: Both 'Sales' and '#Order' distributions are right-skewed with a significant number of outliers, indicating that a small portion of transactions account for a large share of sales and orders. There is a very strong positive correlation (0.94) between '#Order' and 'Sales', confirming that more orders directly translate to higher sales.

2.  **Impact of Discounts**: Discounts significantly increase mean sales. Mean sales on days with discounts ($49,426.50) are substantially higher than on days without discounts ($37,403.68). This suggests that discounts are an effective tool for boosting revenue, although it's important to analyze profitability per transaction.

3.  **Holiday Effect**: Counterintuitively, mean sales on holidays ($35,451.88) are lower than on non-holidays ($43,897.29). This could indicate that customers are either stocking up before holidays or that certain holidays lead to store closures/reduced hours, thus affecting sales volume. Further investigation into specific holiday types and their impact is warranted.

4.  **Regional and Store Type Variations**: Significant differences exist in median sales across regions and mean sales across store types. Region R1 has the highest median sales ($43,125.00), while Store Type S4 generates the highest mean sales ($59,945.69), significantly outperforming Store Type S2 ($27,530.83). This highlights the importance of localized strategies.

5.  **Time Series Patterns**: Daily sales exhibit fluctuations over time, suggesting underlying seasonality or trends that require robust forecasting models.

6.  **Outliers**: A substantial number of outliers in both 'Sales' (5843) and '#Order' (7089) indicates frequent occurrences of exceptionally high transactions. These could be special events, bulk purchases, or promotional days, which are crucial for accurate forecasting and should be handled appropriately by models.

#### Recommendations:

1.  **Strategic Discounting**: Given the positive impact of discounts on sales, refine discounting strategies by analyzing which products or categories respond best to promotions and during which periods. Implement targeted campaigns to maximize sales and monitor the profitability of discounted items.

2.  **Holiday Season Optimization**: Investigate the reasons behind lower holiday sales. If it's due to reduced operating hours, consider adjusting them or implementing pre-holiday promotions. If customer behavior shifts, develop specific holiday marketing and inventory strategies (e.g., focusing on specific holiday-related products or online sales).

3.  **Tailored Regional and Store Type Strategies**: Develop region-specific and store-type-specific sales targets, marketing campaigns, and inventory management plans. Allocate resources disproportionately to higher-performing store types (like S4) and regions (like R1) or devise strategies to boost sales in underperforming segments (like S2).

4.  **Inventory Optimization**: Leverage the strong correlation between orders and sales to optimize inventory levels. Accurate sales forecasts will directly aid in minimizing stockouts for high-demand products and reducing overstocking for lower-demand items, thereby enhancing supply chain efficiency.

5.  **Model Implementation**: The **XGBoostRegressor** demonstrated the best performance among all models tested, achieving the lowest RMSE (0.4937) and highest R-squared (0.7621) on the scaled validation data. It is recommended for deployment. The Ensemble model, while slightly behind, also showed strong performance and could be considered for increased robustness.

    *   **Further Improvement**: Continue with hyperparameter tuning for XGBoost and LightGBM models to potentially achieve even better performance. Explore more advanced feature engineering, such as interactions between features or external data (e.g., local events, weather, economic indicators), to capture more complex patterns.

6.  **Outlier Management**: Implement robust outlier detection and handling mechanisms. Instead of simply removing them, analyze their root causes and potentially model them as special events (e.g., using a separate model or a flag in the main model) to prevent loss of valuable sales information.

By integrating these insights and implementing the recommended strategies, the business can significantly enhance its sales forecasting capabilities, leading to more informed decision-making in inventory, resource allocation, and marketing.

### Comparison of model performance and conclusions
This section reviews and compares the performance of all implemented models on the validation set. The comparison_df provides a structured overview of each model's effectiveness, sorted by their Root Mean Squared Error (RMSE).

**Model Performance Metrics (Sorted by RMSE):**
$Model         MAE           MSE        RMSE  R-squared  \
5        XGBoostRegressor    0.341654      0.243767    0.493728   0.762109   
7  Ensemble (RF+XGB+LGBM)    0.354917      0.258409    0.508339   0.747820   
6      LightGBM Regressor    0.362230      0.267868    0.517560   0.738589   
4   RandomForestRegressor    0.412658      0.344434    0.586885   0.663868   
0       Linear Regression    0.464298      0.446646    0.668316   0.564120   
3                 Prophet  113.844120  25829.294776  160.714949   0.191527   
2                  SARIMA  130.238581  30869.773852  175.697962   0.033757   
1                   ARIMA  134.983148  34768.548900  186.463264  -0.088277   
         MAPE  
5         NaN  
7         NaN  
6         NaN  
4         NaN  
0  365.365582  
3  363.903054  
2  339.661375  
1  280.415772$
**Analysis and Conclusions**

The comparison table above highlights a clear distinction in performance between the traditional machine learning regression models (Linear Regression, RandomForestRegressor, XGBoostRegressor, LightGBM Regressor) and the time-series specific models (ARIMA, SARIMA, Prophet).

**Best-Performing Model:**

The XGBoostRegressor stands out as the top performer with the lowest RMSE (0.4937), lowest MAE (0.3417), and highest R-squared (0.7621). Its ability to accurately capture the complex relationships within the data makes it the most suitable model for this sales forecasting task.

**Performance of Tree-Based Ensemble Models:**

Tree-based ensemble models (XGBoost, LightGBM, and RandomForest) consistently outperformed Linear Regression and significantly outshone the time-series models. This superiority can be attributed to several factors:

**Non-linear Relationships:** Sales data often exhibit complex, non-linear interactions between various features (e.g., store type, promotions, time of year). Tree-based models are inherently capable of modeling these non-linearities and intricate feature interactions, which simpler linear models cannot.
**Feature Engineering:** The extensive feature engineering performed (e.g., Year, Month, DayOfWeek, Sales_Rolling_Mean_7D, one-hot encoded categorical features) provided these models with rich contextual information beyond just the raw sales figures. Ensemble methods excel at leveraging such diverse feature sets to make more informed predictions.
**Robustness to Outliers:** Ensemble models, particularly gradient boosting methods like XGBoost and LightGBM, are generally more robust to outliers and noise in the data, which were identified as prevalent in the Sales and #Order distributions during EDA.
**Discrepancy in Error Metrics (Absolute vs. Scaled Values):**

It is crucial to note the large difference in the magnitude of MAE, MSE, and RMSE values between the regression models and the time-series models. This is primarily due to a difference in data scaling:

**Regression Models:** The ‘Sales’ column (target variable y_train) was scaled using StandardScaler before training the Linear Regression, RandomForest, XGBoost, and LightGBM models. This transformed the Sales values to have a mean of 0 and a standard deviation of 1. Therefore, their error metrics are also on a scaled, unit-less basis, typically small fractional values.
**Time-Series Models:** For ARIMA, SARIMA, and Prophet, the models were trained on the original, unscaled daily_sales data. Consequently, their MAE, MSE, and RMSE values are in the original currency units (e.g., dollars), explaining their much larger numerical values.
Direct comparison of these absolute error metrics (MAE, MSE, RMSE) between the two groups of models is therefore misleading. While the MAPE (Mean Absolute Percentage Error) aims to provide a more comparable relative error by expressing error as a percentage, its calculation was omitted for the tree-based models and the linear regression, limiting a full direct comparison on this specific metric. However, within each group (scaled regression models vs. unscaled time-series models), the metrics are comparable.

**Ensemble Model Performance:**

The simple averaging ensemble of RandomForest, XGBoost, and LightGBM performed very strongly, coming in second place with an RMSE of 0.5083 and R-squared of 0.7478. While it didn’t surpass the best individual model (XGBoost) in terms of RMSE, it demonstrates the potential for combining models to potentially improve robustness or slightly enhance accuracy in different scenarios.

**Conclusion and Recommendations:**

Based on the comprehensive evaluation, the XGBoostRegressor is the recommended model for sales forecasting in this project due to its superior performance on the validation set. Its robustness and ability to capture complex data patterns make it an excellent choice for predicting future sales.

**Further steps could include:**

**Hyperparameter Tuning:** Optimize the hyperparameters of the XGBoostRegressor (and potentially LightGBM) to squeeze out even more performance.
**Feature Importance Analysis:** Utilize the feature importance capabilities of tree-based models to gain deeper insights into which factors most influence sales.
**Cross-Validation:** Implement more rigorous cross-validation strategies, especially time-series cross-validation for the time-series models, to ensure the robustness of the performance metrics.
**Monitoring and Retraining:** Establish a pipeline for continuous monitoring of the model’s performance in production and periodic retraining with new data to maintain accuracy.

### Deployment Steps (with Flask API)

**XGBoostRegressor** emerged as the best overall model, achieving the lowest RMSE and highest R-squared on the validation set. This indicates its superior accuracy and ability to explain the variance in sales data compared to other models.

### Conceptual Deployment Steps (with Flask API)

To deploy the chosen best model, the **XGBoostRegressor**, using a Flask API, the following conceptual steps would be involved:

1.  **Model Export (Persistence)**:
    *   The trained `XGBoostRegressor` model (`model_xgb`) would be saved to a persistent file format (e.g., `.pkl` or `.joblib`) using libraries like `pickle` or `joblib`. This allows the model to be loaded and used for predictions without retraining.
    *   `import joblib; joblib.dump(model_xgb, 'xgboost_sales_model.pkl')`

2.  **API Development (Model Serving with Flask)**:
    *   A RESTful API would be built using the Flask web framework to expose the model's prediction capabilities.
    *   **Loading**: The saved `xgboost_Flask.py` would be loaded into memory when the Flask application starts, ensuring efficient access during inference.
    *   **Endpoint**: A dedicated endpoint, for example, `/XGBoost_predict`, would be created (e.g., using `@app.route('/XGBoost_predict', methods=['POST'])`). This endpoint would listen for incoming HTTP POST requests.
    *   **Input Handling**: The API endpoint would receive input features (e.g., JSON data corresponding to `X_test` columns) from client applications. Robust input validation would be implemented to ensure data quality and format consistency.
    *   **Preprocessing**: Crucially, any data preprocessing steps applied during model training (e.g., scaling numerical features with `StandardScaler`, one-hot encoding categorical features with `pd.get_dummies`) **must** be reapplied to the incoming request data within the Flask application before passing it to the loaded XGBoost model.
    *   **Prediction**: The preprocessed input would then be fed to the loaded `model_xgb_final` to generate sales predictions.
    *   **Output**: The API would return the prediction(s) to the client, typically in a JSON format. If the target variable ('Sales') was scaled during training, an inverse transformation would be applied here to return sales in their original magnitude.
    *   **Authentication**: To secure access, API keys or token-based authentication mechanisms would be integrated into the Flask application.

