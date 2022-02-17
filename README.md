# Rossmann Store Sales -> Data Science Project

![rossmann](https://user-images.githubusercontent.com/75986085/152585675-a7ceff53-8a6f-4548-84ea-abfd32c00fbf.png)


<h2>0. Rossmann Stores Data and Info</h2>
<p>Rossmann are a chain of pharmacies located in Europe, mainly in Germany, with around 56,200 employees and more than 4000 stores. The company was founded by Dirk Rossmann with its headquarters in Burgwedel near Hanover in Germany. ~ Wiki.</p>
<p>What is a pharmacies chain?<br>Basically the chain start with one shop open named 'matrix', the next shop open is the branche. Have two types of chain, the associated chain is when several pharmacies with different owners get together & private chain which starts with the 'matrix' company and has a general owner.</p>

<p>Rossmann CFO on a monthly results meeting asked to all store mananger a sales forecast for the next six monsths.</p>

<p>CFO Like to know next sales for start a reform of all shops.
Pharmaceutical Bussiness Model
Rossmann is present with an e-commerce and in physical stores available for sales of household items, makeup and of course drugstore items, as it is a chain of pharmacies, it is spread over several parts of Europe, thus being able to select regions with greater growth potential and reducing the competition rate.
'First Assumptions'</p>

<ul>
  <dl>
    <dt>Market Size.</dt>
      <dd>All persons over 18 years of age, with preference for older persons.</dd>
    <dt>Marketing Channels.</dt>
      <dd>Rossmann Website & Shops.</dd>
    <dt>Principal Metrics.</dt>
      <dd>Channel Offline: Working on physical stores.</dd>
      <dd>Recency: Purchases over time.</dd>
      <dd>Frequency: Shop sales frequency for sales forecast.</dd>
      <dd>Market Share: Sales competitions.</dd>
  </dl>
</ul>

![chanel](https://user-images.githubusercontent.com/75986085/153439927-f4684894-7067-4023-b089-2116e6d5a7bb.png)

1. Do older customers buy more from physical stores or from competitors?
2. What is the marketing investment compared to physical stores in terms of e-commerce?
3. What are the new products that make customers buy from Rossmann stores instead of competing stores?
4. How do these stores behave in terms of receiving new merchandise to sell to new customers?
5. Are the products sold easily accessible?
6. How are the prices of the products in relation to the location of the stores?
7. How are rossmann products and stores being evaluated?
8. What is the buying process like for these customers?
9. Would a customer who bought in a physical store buy again?
10. How much does a customer cost for physical stores?
11. Who are the main partners of the rossmann brand?
12. Is there a community of customers where they can engage with the products? ...

<p>Data Information at: https://www.kaggle.com/c/rossmann-store-sales</p>

<h2>1. Bussines Problem</h2>

<p>Rossman's CFO would like to predict how much money its stores will generate to renovate them in the future.</p>
<p>Rossmann CFO, asked to all of shops merchant's to send for him this prediction, with this problem, all rossmann's merchant's asked to data/analisys team this prediction.</p>
<p>New Version of project (04/02/2021)</p>
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
    <dt>Data Preparation.</dt>
      <dd>Split & Prepare and Prepare & Split, this two versios of preparation can provide data leak.</dd>
    <dt>Machine Learning Modeling.</dt>
      <dd>Selection of Four ML Models, Base, Linear and two Tree-Based.</dd>
  </dl>
</ul>

<h2>3. EDA Insight's</h2>

<p>After brainstorming and hypothesis validation, some insights appeared.</p>
<h3> Top 3 Insight's </h3>
<ul>
  <li>Stores with large assortment, sell less.</li>

![sales](https://user-images.githubusercontent.com/75986085/153096505-fe9a9afb-f6e6-451d-a85d-d5579839071d.jpeg)

  
  <li>Stores with consecutive promo, sell less if long time of promo.</li>
  
![promo](https://user-images.githubusercontent.com/75986085/153096571-6f01a3b5-a87c-487d-acdd-7c1592e379c3.jpg)

  
  <li>Stores with closely competitors, sell more.</li>
  
![less](https://user-images.githubusercontent.com/75986085/153096584-eb58b3c4-2d4e-457e-a7f8-82ef6f9b5604.jpg)
  
</ul>

<h2>4. Data Preparation</h2>
<p>When you have "date" features on dataset, its possible to you get data leak during model training. I have selected two tipes of preparation, one splited train and test after data preparation, and before data preparation to check the data leakege.</p>
<ul>
  <dl>
    <dt>Categorical Data.</dt>
      <dd>Used the Frequency Encoding to all Categorical Data.</dd>
    <dt>Rescaling.</dt>
      <dd>After KStest and QQplot, it was not necessary to normalize.</dd>
    <dt>Data Filtering.</dt>
      <dd>Simple way to reduce dimensionality of dataset.</dd>
    <dt>Exploration Data Analysis.</dt>
      <dd>Validation of all hypotesis list with data.</dd>
    <dt>Data Preparation.</dt>
      <dd>Split & Prepare and Prepare & Split, this two versios of preparation can provide data leak.</dd>
    <dt>Machine Learning Modeling.</dt>
      <dd>Selection of Four ML Models, Base, Linear and two Tree-Based.</dd>
  </dl>
</ul>
