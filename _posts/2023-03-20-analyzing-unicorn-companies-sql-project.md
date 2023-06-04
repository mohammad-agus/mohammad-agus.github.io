---
layout: post
title: "Analyzing Unicorn Companies"
info: #
tech: "sql"
type: mini project, data transformations
---



Analyzing Unicorns Companies is a DataCamp unguided SQL project that intends to find the number of companies that achieved unicorn status (values exceeding USD 1 billion) between 2019 and 2021. Check out [my GitHub repository](https://github.com/mohammad-agus/analyzing_unicorn_companies_sql_project) or [project page](https://app.datacamp.com/learn/projects/1531) to download the datasets.


# Tables
Given is the `unicorns` database, which contains the following tables, for this project's dataset sources. 

* `dates` table

| Column       | Description                                  |
|------------- |--------------------------------------------- |
| company_id   | A unique ID for the company.                 |
| date_joined  | The date that the company became a unicorn.  |
| year_founded | The year that the company was founded.       |



```sql
SELECT * FROM dates
LIMIT 5;
```

![Expected result](/assets/2023_analyzing_unicorn_companies/dates.png)



* `funding` table

| Column           | Description                                  |
|----------------- |--------------------------------------------- |
| company_id       | A unique ID for the company.                 |
| valuation        | Company value in US dollars.                 |
| funding          | The amount of funding raised in US dollars.  |
| select_investors | A list of key investors in the company.      |



```sql
SELECT * FROM funding
LIMIT 5;
```

![Expected result](/assets/2023_analyzing_unicorn_companies/funding.png)





* `industries` table

| Column       | Description                                  |
|------------- |--------------------------------------------- |
| company_id   | A unique ID for the company.                 |
| industry     | The industry that the company operates in.   |




```sql
SELECT * FROM industries
LIMIT 5;
```

![Expected result](/assets/2023_analyzing_unicorn_companies/industries.png)



* `companies` table

| Column       | Description                                       |
|------------- |-------------------------------------------------- |
| company_id   | A unique ID for the company.                      |
| company      | The name of the company.                          |
| city         | The city where the company is headquartered.      |
| country      | The country where the company is headquartered.   |
| continent    | The continent where the company is headquartered. |



```sql
SELECT * FROM companies
LIMIT 5;
```

![Expected result](/assets/2023_analyzing_unicorn_companies/companies.png)



# Tasks
* Identify the three best-performing industries based on the number of new unicorns created between 2019 and 2021 combined.
* Write a query to return the industry, the year, and the number of companies in these industries that became unicorns each year in 2019, 2020, and 2021, along with the average valuation per industry per year, converted to billions of dollars and rounded to two decimal places!
* Display the result by industry, then the year in descending order.


The final output of the query will look like this:
<br/>

![Expected result](/assets/2023_analyzing_unicorn_companies/expected_result.png)

Where industry1, industry2, and industry3 are the three top-performing industries.
<br/>

# Query and Result


```sql
SELECT
	i.industry,
	EXTRACT(YEAR FROM d.date_joined) AS year,
	COUNT(c.company_id) AS num_unicorns,
	ROUND(AVG(f.valuation/1000000000), 2) AS average_valuation_billions
FROM companies c
	-- inner join companies with industries, dates and funding table
	INNER JOIN industries i USING(company_id)
	INNER JOIN dates d USING(company_id)
	INNER JOIN funding f USING(company_id)
WHERE
	-- extract the year from d.date_joined, then filter the years 2019 - 2021
	EXTRACT(YEAR FROM d.date_joined) BETWEEN 2019 AND 2021
	
	-- filter the top 3 industries based on the number of new unicorns
	 AND i.industry IN (
		SELECT i.industry
		FROM 
			companies c
			INNER JOIN industries i ON c.company_id = i.company_id
			INNER JOIN dates d ON c.company_id = d.company_id
		WHERE
			EXTRACT(YEAR FROM d.date_joined) BETWEEN 2019 AND 2021
		GROUP BY 
			i.industry, 
			EXTRACT(YEAR FROM d.date_joined)
		ORDER BY 
			COUNT(c.company_id) DESC
		LIMIT 3)

GROUP BY 
	i.industry,
	EXTRACT(YEAR FROM d.date_joined)
ORDER BY
	i.industry,
	year DESC;

```

![Expected result](/assets/2023_analyzing_unicorn_companies/result.png)