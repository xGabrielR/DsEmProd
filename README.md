# Rossman Store Sales -> Data Science Project

![Sem título](https://user-images.githubusercontent.com/75986085/126849210-c38edf19-e6f5-47a5-b008-369afd0c6f87.png)

<h2>1. Bussines Problem</h2>

<p>Rossman's CFO would like to predict how much money its stores will generate to renovate them in the future.</p>
<p>Rossmann CFO, asked to all of shops merchant's to send for him this prediction, with this problem, all rossmann's merchant's asked to data/analisys team this prediction.</p>

<h2>2. Solution Strategy & Assumptions </h2>
<h3>First CRISP Cycle</h3>

<ul>
  <dl>
    <dt>Data Clearing & Descriptive Statistical.</dt>
      <dd>First real step is download the dataset, import in jupyter and start in seven steps to change data types, data dimension, fillout na... At first statistic dataframe, i used simple statistic descriptions to check how my data is organized.</dd>
    <dt>Feature Engineering.</dt>
      <dd>In this step, with coggle.it to make a mind map and use the mind map to create some hypothesis list, after this list, i created some new features based on date.</dd>
    <dt>Data Filtering.</dt>
      <dd>Simple way to reduce dimensionality of dataset.</dd>
    <dt>Exploration Data Analysis.</dt>
      <dd>Validation of all hypotesis list with data.</dd>
  </dl>
</ul>

<h2>3. EDA Insight's</h2>

<p>After brainstorming and hypothesis validation, some insights appeared.</p>
<h3> Top 3 Insight's </h3>
<ul>
  <li>Stores with large assortment, sell less.</li>
  <li>Stores with consecutive promo, sell less.</li>
  <li>Stores with closely competitors, sell more.</li>
</ul>

<h2>4. Machine Learning Model</h2>

<p>Selected have used 5 Models.</p>
<ul>
  <li>Linear Regression.</li>
  <li>Bayesian Regression.</li>
  <li>Regularized Linear Regression.</li>
  <li>Random Forest Regressor.</li>
  <li>XGBoost Regressor.</li>
</ul>

![1](https://user-images.githubusercontent.com/75986085/129580857-8522b49a-2a36-40d2-a658-bab068278511.png)

<p>Used XGBOOST in Hyperparameter.</p>

<h2>4.x Neural Network</h2>
<p>After completed the first cycle, i have tryied to use a neural network.</p>

<h2>5. Model Evaluatin</h2>

<ul>
  <dl>
    <dt>Geral Forecast.</dt>
      <dd>R$ 285,633,248.00</dd>
    <dt>Worst Scenario.</dt>
      <dd>R$ 284,899,971.45</dd>
    <dt>Best Scenario.</dt>
      <dd>R$ 286,366,504.10.</dd>
  </dl>
</ul>

<h2>6. Second Cycle</h2>
<p>In second cycle, is possible to add new features, use school holiday in prediction and optimize neural network.</p>
<p>Is possible to know more about another stores, to add new features, because have a little stores without a good model forecast.</p>
<p>Any other ideia, is deploy with streamlit, CFO need to see dashboards too.</p>
