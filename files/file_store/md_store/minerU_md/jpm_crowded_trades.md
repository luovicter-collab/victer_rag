# the journal of PORTFOLIO management

![](images/c0fbd426cd681ec9f5f1dc281c9bfe52aef38080e9ecdd153f6ecbf3fe445bb6.jpg)

volume 45

number 5

ULY 2019

jpm.iprjournals.com

Crowded Trades: Implications for Sector Rotation and Factor Timing

William Kinlaw , Mark Kritzman, and David Turk ington

# Crowded Trades: Implications for Sector Rotation and Factor Timing

William Kinlaw , Mark Kritzman, and David Turk ington

# William Kinlaw

is a senior managing director at State Street Associates in Cambridge, MA. wbkinlaw@statestreet.com

# Mark Kritzman

is chief executive officer of Windham Capital Management and senior lecturer at MIT’s Sloan School of Management in Cambridge, MA. kritzman@mit.edu

# David Turk ington

is a senior managing director at State Street Associates in Cambridge, MA. dturkington@statestreet.com

*All articles are now categorized by topics and subtopics. View at IPRJournals.com.

ABSTRACT: Crowded trades are often associated with bubbles. If investors can locate a bubble sufficiently early, they can profit from the run-up in prices. But to profit from a bubble, investors must exit the bubble before the sell-off erodes all of the profits. The authors propose two measures for managing exposure to bubbles. One measure, called asset centrality, locates crowded trading, which they show is often associated with the formation of bubbles. The other is a measure of relative value, which helps to separate crowding that occurs during a bubble’s run-up from crowding that occurs during a bubble’s sell-off. Neither measure by itself is sufficient for identifying the full cycle of a bubble, but the authors show that together these measures have the potential to locate bubbles in sectors and in factors as they begin to emerge and to identify exit points before they fully def late.

TOPICS: Portfolio construction, factors, risk premia, risk management*

rowded trading refers to the deployment of a large amount of capital to purchase or sell an asset or a group of assets with similar characteristics, which can result in a significant change in the price of the assets. It is important to note, however, that not all large price moves are caused by crowded trading. If investors perceive a change in the fundamental value of an asset, its price should adjust

whether or not there is crowded trading. However, when prices change significantly from a shift in investor psychology or infatuation with a new strategy, it is likely due to crowded trading. This distinction is critical. Price adjustments that correspond to shifting fundamentals should persist indef initely, which makes them difficult to exploit, but price adjustments that occur in the absence of a shift in fundamentals are more likely to be transitory and, therefore, more easily exploitable. This type of crowded trading, in which asset prices rise and fall significantly without a commensurate change in fundamental value, is commonly referred to as a bubble. Our goal in this article is to demonstrate how to locate bubbles and to determine whether they are inf lating or def lating.

The academic community is undecided about whether it is possible to detect bubbles before they crash. For example, in his Nobel Prize acceptance speech, Fama (2014) stated: “The available research provides no reliable evidence that stock market price declines are ever predictable.” In a Chicago Booth Review interview, Fama (2016) added: “Statistically, people have not come up with ways of identifying bubbles.” In response to this challenge, Greenwood, Shleifer, and You (2017) evaluated 40 US and 107 international bubbles, which they defined as events in which prices increased by at least $100 \%$ both on an absolute and relative-to-market

basis. They found that price increases alone do not reliably predict negative expected returns, though they do predict a larger probability of loss. The authors also found that variables such as increased volatility, new stock issuance, and price acceleration have the potential to differentiate price increases that crash from those that continue to rise.

We present a different approach for predicting bubbles than Greenwood, Shleifer, and You. We use asset centrality as a proxy for crowded trading to locate bubbles, and we combine this signal with a measure of relative value to distinguish crowded trading that occurs during a bubble’s run-up from crowded trading that occurs during its sell-off. We also use a somewhat loose def inition of a bubble. Whereas bubbles are typically associated with a spectacular run-up and sell-off, we def ine a bubble as any signif icant increase and decrease in prices that is not associated with a corresponding change in fundamentals. Along these lines, Cornell (2018) argued that small bubbles and crashes should be very common in markets, possibly as a result of limits to arbitrage or trading that occurs for any reason other than an assessment of fair value.

That said, we offer results that contradict Fama and extend the findings of Greenwood, Shleifer, and You. We show that investors could have profited by overweighting sectors within the US equity market that were crowded but not overvalued and by underweighting sectors that were crowded and simultaneously overvalued. This result extends as well to other equity markets that we tested. We also produce results showing that investors could have profited by deploying the same strategy to manage exposure to well-known equity factors.

We begin by describing asset centrality, and we offer conjectures about why it may be associated with crowded trading. We then describe a relative value measure that we use to separate crowding that occurs during a bubble’s run-up from crowding that occurs during its sell-off. Next, we examine some well-known bubbles, and we show how our centrality and valuation measures evolved over the course of these bubbles. Then we apply our measures out of sample to identify bubbles among sectors within the US stock market. We also apply our methodology to other equity markets and to well-known factors in the US equity market. We present

persuasive evidence of the efficacy of our methodology. We conclude with a summary.

# ASSET CENTRALITY

In our analysis, we do not directly observe crowded trading from f low information. Instead, we infer it from price behavior. We use a measure called asset centrality, which captures the extent to which a group of similar assets drives the variability of returns across a broader universe of which it is a member.

We begin by calculating the covariance matrix of a set of sectors.1 We then perform a principal components analysis to identify the eigenvectors that explain the variability of returns across the sectors. Next we compute the absorption ratio, which equals the fraction of total variance that is explained, or absorbed, by a fixed number of eigenvectors, as shown in Equation 1.

$$
A R = \frac {\sum_ {i = 1} ^ {n} \sigma_ {E i} ^ {2}}{\sum_ {j = 1} ^ {N} \sigma_ {A j} ^ {2}} \tag {1}
$$

In Equation 1, $N$ equals the number of sectors, n equals the number of eigenvectors in the numerator of absorption ratio, ${ \sigma } _ { E i } ^ { 2 }$ equals the variance of the ith eigenvector, and ${ \sigma } _ { A j } ^ { 2 }$ equals the variance of the jth sector. The absorption ratio is commonly used to measure risk concentration.2

Finally, we define centrality by noting a given sector’s weight (as an absolute value) in each of the eigenvectors that comprise the subset of the most important eigenvectors (those in the numerator of the absorption ratio). We then multiply the sector’s weight in each eigenvector by the relative importance of the eigenvector, which we measure as the percentage of variation explained by that eigenvector divided by the sum of the percentage of variation explained by all of the eigenvectors comprising the subset of the most important ones. This gives a measure of centrality, which is defined in Equation 2.3

$$
C _ {i} = \frac {\sum_ {j = 1} ^ {n} \left(A R ^ {j} \cdot \frac {\left| E V _ {i} ^ {j} \right|}{\sum_ {k = 1} ^ {N} \left| E V _ {k} ^ {j} \right|}\right)}{\sum_ {j = 1} ^ {n} A R ^ {j}} \tag {2}
$$

In Equation 2, $C _ { i }$ equals a sector’s centrality score, $A R ^ { j }$ is the absorption ratio of the jth eigenvector, $E V _ { i } ^ { j }$ equals the absolute value of the exposure of the ith sector within the jth eigenvector, n equals the number of eigenvectors in the numerator of the absorption ratio, and $N$ equals the total number of sectors.4

To lend some intuition to centrality, think about this computation using only the first eigenvector as the numerator in the absorption ratio. This special case would be very similar to the technique used in Google’s PageRank algorithm (see Brin and Page 1998). Think of each asset group as a node on a map. By defining an importance score as the sum of all of its neighbors’ importance scores (times some constant), that score would equal precisely the weight of the sector in the first eigenvector.5 We instead use several eigenvectors in the numerator of the absorption ratio because most of the time several factors contribute importantly to asset variability.

Why might centrality be indicative of crowded trading? First, consider the features that contribute to a sector’s centrality. Centrality is higher for sectors that are relatively more volatile and for sectors that are more connected to other sectors.

Now consider how crowded trading affects volatility and connectivity. Crowded trading raises a sector’s

volatility in two ways. First, when a large quantity of capital is deployed to purchase or sell a particular sector, it leads to large order imbalances, which result in large price adjustments and, hence, greater volatility. Second, the components within the sector become more correlated with each other as investors focus their trades on the sector as a single unit rather than distribute their trades independently across the components of the sector. Volatility therefore rises as the sector becomes less diversified. We would also expect to see an association between crowded trading and connectivity. As investors crowd into sectors, these sectors become bellwethers and, therefore, drive the behavior of other sectors, which raises their connectivity.

Exhibit 1 presents evidence that high centrality coincides with crowded trading. It is a heat map of the cross-sectional percentile ranks of centrality through time for the major sectors of the US equity market. Centrality rises as the colors shift from dark blue to light blue to green to yellow to orange to light red to dark red.6

Notice that the centrality of the tech sector is very high during the tech bubble and that centrality is also very high for the f inancial sector during the global f inancial crisis and very high for the energy sector during periods of volatile energy prices. Also note that for decades the centrality of REITs (real estate investment trusts) was the lowest of any sector, but it rose noticeably during the housing bubble that led to the f inancial crisis.

Exhibit 1 offers persuasive evidence that high centrality corresponds with asset price bubbles, but in order to profit from bubbles, we must be able to time them. Unfortunately, centrality by itself is of little value for timing exposure to bubbles because, although it rises as a bubble begins to emerge, it continues to rise throughout the entire bubble cycle and often more steeply during a bubble’s sell-off. We, therefore, introduce a measure of relative value to separate centrality that occurs during a bubble’s run-up from centrality that occurs during its sell-off.

![](images/0335ffa8d91acf4bea1a1ad3a9c7cf6323b8dc2c2c04fce7e740ba98a54929ee.jpg)  
E x h i b i t 1 Heat Map of Sector Centrality

# RELATIVE VALUE

We use the price-to-book-value ratio to compute relative value. We f irst calculate each sector’s priceto-book-value ratio. Then we normalize each sector’s valuation by dividing it by its own 10-year average to account for the fact that some sectors have consistently higher valuations than other sectors. Finally, we divide each sector’s normalized valuation by the average of the normalized valuations of all the other sectors to arrive at a cross-sectional measure of relative value.

This measure of relative value serves the following purpose. As crowding occurs within a particular sector, the centrality of the sector rises, along with its relative value. It is the rise in centrality that signals the emergence of a potential bubble. As crowding continues, centrality and relative value continue to rise until some news triggers a shift in psychology, at which point investors begin to exit the trade. As this occurs, centrality continues to rise because crowding persists but now in the opposite direction. Indeed, crowding is usually more intense when investors exit a trade than when they enter it. This crowded selling drives down the relative value of the group, typically early during a bubble’s sell-off or at least before the sell-off has run its course.

One might be tempted to focus only on relative value to time exposure to bubbles, but relative value by itself is inadequate because it cannot distinguish price activity that is driven by shifting fundamentals from price activity that is driven by changes in investor

psychology. We therefore use centrality to distinguish bubbles from fundamentally justified price increases and relative value to distinguish centrality that occurs as a bubble inf lates from centrality that occurs during a bubble’s sell-off.

# CENTRALITY AND RELATIVE VALUE DURING WELL-KNOWN BUBBLES

We now show how our measures of asset centrality and relative value evolved during two well-known bubbles, the tech bubble and the housing bubble.

The tech bubble occurred from 1998 through 2000 as investors became infatuated with technology stocks. Exhibit 2 compares the evolution of stock prices with the evolution of centrality and relative value. Panel A of Exhibit 2 shows the run-up and sell-off of tech stock prices during the bubble. Panel B reveals that the level of centrality increased significantly during the run-up phase of the bubble and then spiked sharply during the precipitous sell-off.7 Panel C shows that relative value followed the same pattern as prices. The vertical dotted line locates the point at which the bubble burst.

Exhibit 3 documents the evolution of centrality and relative value during the housing bubble that led to the credit bubble and eventually to the global financial crisis. As was the case with the tech bubble, centrality

# E x h i b i t 2

# The Tech Bubble

![](images/7b37f5ba2685dd59ae455d497cb6799e69c9fc2afe6f2769327d63ddf14e6db6.jpg)  
3DQHO$7HFK6HFWRU3ULFH

![](images/6e12f3a81c9a9186cb7f5ee96620882f29d03211aa3b680e576b046ce877ae23.jpg)  
3DQHO%&HQWUDOLW\

![](images/8518815df5765b85607e7871b7498ca31794b5c2704636a87c112286a4db3575.jpg)  
3DQHO&5HODWLYH9DOXH

# E x h i b i t 3

# Housing Bubble

![](images/1193b26770c4e96fbe90956df4fcdcde98c50ec9321febc83c04a26e4c2839c6.jpg)  
3DQHO$5(,76HFWRU3ULFH

![](images/77874a0a634ca9eb2733c7b46033ef752133f953af2c7b073d898e040acc9ad3.jpg)  
3DQHO%&HQWUDOLW\

![](images/a466290b4d48ee0819b4bc94fef2ad8a21a57feeca24133459bfa7c96d719af0.jpg)  
3DQHO&5HODWLYH9DOXH

offered an excellent signal that the bubble was under way, and relative value separated the run-up from the sell-off.

We also examined the portfolio insurance bubble, the credit bubble, and the risk parity bubble. The portfolio

insurance bubble occurred as institutional investors channeled huge amounts of capital into dynamic trading strategies designed to mimic the payoff pattern of a protective put option applied to the S&P 500 Index. Investors believed this strategy would protect them from stock

market losses greater than $5 \%$ ,8 so they invested in stocks more aggressively than they might have otherwise. In mid-October 1987, the stock market responded negatively to news that Congress would seek to limit interest deductions on debt used to finance acquisitions. The portfolio insurance rules then triggered more selling, which fed upon itself, causing the stock market to crash more than $20 \%$ in a single day.

The credit bubble arose from the housing bubble as financial institutions packaged mortgages into securities that found their way to the balance sheets of various organizations.

The risk parity bubble occurred from 2011 to 2013, when many institutional investors allocated large sums of capital to a strategy in which each major component of the portfolio was sized so that it contributed equally to the portfolio’s overall risk. The rationale for this strategy was that low-volatility assets such as bonds had higher return-to-risk ratios than higher-volatility assets such as stocks. In the case of all of these bubbles, we observed a similar pattern for prices, centrality, and relative value.

These bubble examples suggest anecdotally that centrality has the potential to locate bubbles as they begin to emerge and that relative value has the potential to separate a bubble’s run-up from its sell-off. Based on the evidence provided from these historical bubbles, we next seek to determine if we can deploy these measures in a sector rotation strategy as well as in a strategy for timing exposure to factors. However, we must f irst resolve a serious challenge, which is that these historical bubbles fail to suggest uniform thresholds for centrality and relative value. We address this problem in two ways. First, rather than focus on the absolute level of centrality, we focus on its change in value, measured as a standardized shift. Second, we compare these measures cross-sectionally; hence, we focus on their relation to one another rather than their individual absolute values.

# IMPLICATIONS FOR SECTOR ROTATION

We conjecture that sectors with high centrality relative to other sectors and that are not relatively

overvalued are those that are in a bubble run-up and that sectors with relatively high centrality and that are relatively overvalued are those that are selling off. As we mentioned earlier, we do not limit our definition of a bubble to major price inf lation and def lation. We consider a bubble as any significant increase and decrease in price that occurs without a corresponding shift in fundamentals. Moreover, we are not so ambitious as to seek to identify the precise beginning, peak, and end of all bubbles. Rather, our goal is to identify the run-up and sell-off phases sufficiently early for enough bubbles so as to add significant value to a passive equity strategy. We next describe how we calibrate our measures of centrality and relative value.

# Sector Centrality

1. Each day, we obtain two years of daily historical returns for 11 US sector indexes from Thomson Datastream.   
2. We weight each sector’s historical daily returns by the square root of its market capitalization weights because larger sectors are relatively more important than smaller sectors.   
3. We apply an exponential time decay with a halflife of one year to emphasize recent data and to allow the impact of historical events to recede gradually as the rolling window advances.   
4. We estimate the covariance matrix across sectors and compute centrality scores using Equation 2 with $n = 2$ eigenvectors.   
5. We compute the standardized shift of centrality for each sector by subtracting the prior three-year average centrality score from the current day’s value and dividing this difference by the standard deviation over the prior three-year period.   
6. We move to the next day and repeat.

# Sector Relative Value

1. Each day, we obtain a given sector’s price-to-bookvalue ratio from Thomson Datastream.   
2. We divide the current price-to-book-value ratio by its average over the past 10 years (or 5 years growing to 10 years in the early portion of the sample) to account for the fact that some sectors consistently have higher ratios than others.

E x h i b i t 4 US Equity Conditional Sector Performance Relative to S&P 500 Index (1985–2017)   

<table><tr><td></td><td>Not Crowded</td><td>Crowded</td></tr><tr><td rowspan="3">Not Overvalued</td><td>No Bubble</td><td>Bubble Run-Up</td></tr><tr><td>0.3%</td><td>6.7%</td></tr><tr><td>0.05</td><td>0.70</td></tr><tr><td rowspan="3">Overvalued</td><td>No Bubble</td><td>Bubble Sell-Off</td></tr><tr><td>0.6%</td><td>-6.5%</td></tr><tr><td>0.07</td><td>-0.50</td></tr></table>

3. We divide each sector’s normalized ratio by the average of the normalized ratios of the other 10 sectors to derive a relative normalized priceto-book-value ratio.   
4. We move to the next day and repeat.

As a proof of concept, we construct sector portfolios that are conditioned on centrality and relative value, and we compute their conditional performance.

# Conditional Sector Performance

1. We rank all sectors by their degree of crowding (centrality shift) and relative value scores, as defined previously.   
2. We identify the sectors that rank in the top three by both measures.   
3. We form equally weighted portfolios for four combinations of crowding and relative value:

a. not top three crowded, not top three overvalued (no bubble)   
b. not top three crowded, top three overvalued (no bubble)   
c. top three crowded, not top three overvalued, increase in price over the prior year (bubble run-up)9   
d. top three crowded, top three overvalued (bubble sell-off )

4. We impose a one-day implementation lag and record the next day’s return for each portfolio (or mark as null if no sectors meet the relevant condition).

5. We move to the next day and repeat.

Exhibit 4 shows annualized performance relative to the S&P 500 Index for portfolios composed of sectors that are not crowded and sectors that are crowded when those sectors are not overvalued and when they are overvalued. It reveals that crowding and relative value by themselves do not provide useful information. However, when used in combination, they are particularly potent. Sectors that were not relatively overvalued but were relatively crowded delivered the best performance. This outcome supports our conjecture that these sectors were in bubble run-ups. In contrast, sectors that were relatively overvalued and crowded had the worst performance, which supports our conjecture that these sectors were selling off. The numbers in italics are the return-torisk ratios.10 They indicate, not surprisingly, that bubble sell-offs are more volatile than bubble run-ups.

Next we test a trading rule to determine the extent to which we can capture the spread between the performance of bubble run-ups and sell-offs.

# Trading Rule

1. We assign expected returns of $+ 5 \%$ to sectors identified as run ups, $- 5 \%$ to those identified as selloffs, and zero otherwise, as defined previously.   
2. We estimate the annualized covariance matrix from the prior five years of daily sector returns.   
3. We solve for the mean–variance long only optimal weights assuming a risk aversion coefficient equal to 1.0.   
4. We def ine the portfolio weights to hold for the next day as the average of the optimal weights over the past quarter (to control for turnover).   
5. We impose a one-day implementation lag and record the next day’s return for the portfolio.   
6. We move to the next day and repeat.

Exhibit 5 reveals that this simple trading rule produced a higher annualized return with less risk than the S&P 500 Index, resulting in a much improved returnto-risk ratio. And for investors who focus on relative performance, it delivered a very attractive information ratio.

E x h i b i t 5 Sector Rotation Strategy Performance (1985–2017)   

<table><tr><td></td><td>S&amp;P 500 Index</td><td>Sector Rotation Strategy</td><td>Sector Rotation Relative Performance</td></tr><tr><td>Return</td><td>11.3%</td><td>15.5%</td><td>4.2%</td></tr><tr><td>Risk</td><td>17.3%</td><td>15.6%</td><td>8.5%</td></tr><tr><td>Ratio</td><td>0.66</td><td>1.00</td><td>0.49</td></tr></table>

Exhibit 6 depicts the outperformance of the sector rotation strategy graphically, and it shows the sector weights that generated this performance. Panel A of Exhibit 6 shows that the sector rotation strategy added value throughout most of the backtest period as opposed to in just one or two episodes.

It is worth noting that our test is a rather conservative one. By restricting the portfolio weights to be positive, we limit our ability to underweight small sectors and to overweight large sectors.

We applied the same trading rules to the stock markets of six other major equity markets, using sector data from Datastream. Exhibit 7 shows that this sector rotation strategy would have increased return in all six of the foreign equity markets that we evaluated relative to their local benchmarks, and it would have reduced risk in all but one of these markets.

We should not be surprised or discouraged by the relatively meager results for Australia. Australia has the most concentrated sector distribution. On average, over the backtest period, its Herfindahl index (sum of squared market capitalization weights) equaled 0.24 compared to 0.12 for the United States. Plainly said, Australia presents a very limited opportunity set.

# IMPLICATIONS FOR FACTOR TIMING

We now turn our attention to equity risk factors. We analyze four factors that are prevalent in the literature: size, value, quality, and low volatility. We do not analyze momentum for two reasons. First, the constituents in a momentum portfolio change frequently, which erodes the connection between crowding and security prices. Second, momentum may be a consequence of crowding in other factors. This overlap could confound our analysis of crowding.

We begin by describing how we define factor portfolios and how we compute factor centrality and relative value.

# Factor Portfolios

1. We rank stocks within the S&P 500 US largecap universe according to each of four security attributes:

a. market capitalization (small size)   
b. price-to-book-value ratio (low valuation)   
c. return on equity (high quality)   
d. prior two-year daily return volatility (low volatility)

2. We construct market-capitalization-weighted portfolios for each of 10 deciles, ranked by the chosen attribute, with an equal number of securities in each portfolio.   
3. We impose a one-day implementation lag and record the daily returns for each decile portfolio.   
4. We define the total return of a given factor as the market-capitalization-weighted average of the top two decile portfolios.   
5. We def ine the relative return of the factor as its total return minus the market-capitalizationweighted S&P 500 (scaling up the low volatility factor portfolio to match the expected volatility of the S&P 500).   
6. We select new securities and adjust portfolio weights after one year.

We calculate centrality for each factor independently because, unlike sectors, multiple factors may contain the same stocks. We do not weight the factor decile returns by market capitalization because we do not expect size to relate to factor relevance in the way it does for sectors.

# Factor Centrality

1. For each factor, we obtain two years of daily historical returns for the 10 decile portfolios for the chosen factor.   
2. We apply an exponential time decay with a halflife of one year to emphasize recent data and to allow the impact of historical events to recede gradually as the rolling window advances.

# E x h i b i t 6

# Cumulative Sector Excess Performance and Sector Weights

![](images/3208f503924d85d1fc23d9a1c4e8c82a5e33d50aa3d0483eb441c6a58da96219.jpg)  
3DQHO$&XPXODWLYH([FHVV3HUIRUPDQFHLQ/RJ5HWXUQV

![](images/a61b56c5a71dfac4e2455ea06fe9918ba4e9e4d49b0a147373f4e1fc3af6f81b.jpg)  
3DQHO%6HFWRU5RWDWLRQ6WUDWHJ\:HLJKWV

3. We estimate the covariance matrix across decile portfolios and compute centrality scores using Equation 2 with $n = 2$ eigenvectors.   
4. We sum the centrality scores of the top two deciles that comprise the factor definition.   
5. We compute the standardized shift of centrality for the factor by subtracting the prior three-year average centrality score from the current day’s value and dividing this difference by the standard deviation over the prior three-year period.   
6. We move to the next day and repeat.

# Factor Relative Value

1. For each factor, we compute the market-capitalization-weighted price-to-book-value ratio for the 10 decile portfolios for the chosen factor, based on the price and book value of the constituents stocks in each decile.11

![](images/9716137caff4723626dfeab9be443230ed6c83c994eed85bbd2ed11f7477668b.jpg)  
E x h i b i t 7 Sector Rotation Strategy Excess Return and Risk in Foreign Equity Markets (1985–2017)

2. We divide each decile’s current price-to-bookvalue ratio by its average over the past 10 years (or 5 years growing to 10 years in the early portion of the sample to obtain longer history) to account for the fact that some deciles consistently have higher ratios than others.   
3. We divide the normalized ratio for the decile by the average of the normalized ratios for the other nine deciles.   
4. We repeat for every decile and compute the market-capitalization-weighted average for the top two deciles for each factor.   
5. We move to the next day and repeat.

We compute conditional factor returns just as we did for sectors. We classify those factors that are the two most crowded but not the two most overvalued and that have outperformed the market index on a relative basis over the past year as factors that are in the run-up phase of a bubble. We also classify those factors that are the top two most crowded and two most overvalued as factors that are selling off. Additionally, we perform the factor timing backtest the same way we did for sectors.

Exhibit 8 reveals that our methodology works just as well for factors as it does for sectors. It shows that the

E x h i b i t 8 US Equity Conditional Factor Performance Relative to S&P 500 Index (1995–2017)   

<table><tr><td></td><td>Not Crowded</td><td>Crowded</td></tr><tr><td rowspan="3">Not Overvalued</td><td>No Bubble</td><td>Bubble Run-Up</td></tr><tr><td>-2.4%</td><td>11.4%</td></tr><tr><td>-0.32</td><td>1.22</td></tr><tr><td rowspan="3">Overvalued</td><td>No Bubble</td><td>Bubble Sell-Off</td></tr><tr><td>0.1%</td><td>2.4%</td></tr><tr><td>0.02</td><td>0.34</td></tr></table>

factors we perceive to be in the run-up phase of a bubble outperformed the factors that we perceive to be selling off by $9 . 0 \%$ annualized. It is worth noting that the factors we deemed to be in a sell-off actually outperformed the market but by significantly less than those we deemed to be in a run-up and by less than the outperformance of an equally weighted static factor portfolio. Effectively, we identified factors that sold off relative to their average premium.

Exhibit 9 shows the annualized results of our trading rules applied to factors. There are a few noteworthy observations. First, an equally weighted static exposure to factors outperformed the S&P 500 Index by $3 . 0 \%$ annually, which supports the popular contention that these equity risk factors carry premiums. Second, the factor timing strategy outperformed the S&P 500 Index by $6 . 3 \%$ . This greater outperformance suggests that, even if factors do not deliver risk premiums, it is still possible that factor timing can add value, just as sector rotation adds value even though aggregate sectors, by construction, cannot outperform the market. Third, for those who care more about relative performance, the factor timing strategy produced a significantly higher information ratio than the static factor portfolio.

Exhibit 10 depicts the factor timing performance graphically, along with the factor weights that delivered this performance. It reveals that the factor timing strategy added value throughout most of the backtest period.

# CONCLUSION

We introduced a measure of asset centrality to locate crowded trades that often lead to the formation of bubbles. We also introduced a measure of relative

# E x h i b i t 9

# Factor Timing Strategy Performance (1995–2017)

<table><tr><td></td><td>S&amp;P 500</td><td>Static Factors</td><td>Factor Timing</td><td>Static Factors Relative Performance</td><td>Factor Timing Relative Performance</td></tr><tr><td>Return</td><td>8.6%</td><td>11.6%</td><td>14.9%</td><td>3.0%</td><td>6.3%</td></tr><tr><td>Risk</td><td>18.6%</td><td>20.5%</td><td>20.8%</td><td>3.7%</td><td>5.7%</td></tr><tr><td>Ratio</td><td>0.46</td><td>0.56</td><td>0.71</td><td>0.82</td><td>1.10</td></tr></table>

# E x h i b i t 1 0

# Factor Timing Strategy Excess Performance and Factor Weights

![](images/e2bb4519fcd80c66bdbb313a12663a1651d8b6ff8246a758d193f5f9af60a70a.jpg)  
Panel A: Cumulative Log Returns to Static Factors

![](images/65bf068fbc4b5f174c9aaf475a60884e2262835b6c00d3bcf6a9b38a4570e957.jpg)  
Panel B: Factor Timing Strategy Weights

value to identify when a bubble transitions from its inf lationary phase to its def lationary phase. We showed that centrality, by itself, is insuff icient for managing exposure to bubbles because it continues to rise and, often, more sharply when crowding coincides with a bubble sell-off. Also, relative value, by itself, is inadequate for managing exposure to bubbles because it fails to distinguish price changes that occur in response to changes in fundamental value from those that result from a shift in investor psychology or herding toward a new strategy. We found, however, that these measures used in tandem could identify run-ups and sell-offs of sector and factor bubbles. We produced a backtest showing that investors could have significantly outperformed the US equity market as well as several foreign equity markets by employing a sector rotation strategy based on these measures. We also produced an additional backtest showing that investors could have significantly outperformed the US equity market as well as a static factor portfolio by using these measures to time exposure to factors. We also applied the same methodology used in Exhibits 4 and 8 to US industries and commodities, which produced consistent results. The annualized spread between the run-up and sell-off phases for US industries was $7 . 6 \%$ , and for commodities it was $1 1 . 1 \%$ . These results suggest that our methodology has applications to many other markets and investment universes.

# ACKNOWLEDGMENTS

We thank Robin Greenwood for valuable comments.

# REFERENCES

Bonacich, P. 1972. “Factoring and Weighting Approaches to Status Scores and Clique Identification.” Journal of Mathematical Sociology 2 (1): 113–120.

Brin, S., and L. Page. 1998. “The Anatomy of a Large-Scale Hypertextual Web Search Engine.” Computer Networks and ISDN Systems 30 (1–7): 107–117.

Cornell, B. 2018. “What Is the Alternative Hypothesis to Market Efficiency?” The Journal of Portfolio Management 44 (7): 3–6.   
. 2014. “Two Pillars of Asset Pricing.” American Economic Review 104 (6): 1467–1485.   
. 2016. “Are Markets Efficient?” Chicago Booth Review, http://review.chicagobooth.edu/economics/2016/video/ are-markets-efficient.   
Greenwood, R., A. Shleifer, and Y. You. 2017. “Bubbles for Fama.” NBER Working paper No. w23191. http://www .nber.org/papers/w23191.   
Kinlaw, W., M. Kritzman, and D. Turkington. 2012. “Toward Determining Systemic Importance.” The Journal of Portfolio Management 38 (4): 100–111.   
Kritzman, M., Y. Li, S. Page, and R. Rigobon. 2011. “Principal Components as a Measure of Systemic Risk.” The Journal of Portfolio Management 37 (4): 112–126.

# Disclaimer

The material presented is for informational purposes only. The views expressed in this material are the views of the authors and are subject to change based on market and other conditions and factors; moreover, they do not necessarily represent the official views of Windham Capital Management, State Street Global Exchange, or State Street Corporation and its aff iliates.

To order reprints of this article, please contact David Rowe at d.rowe@pageantmedia.com or 646-891-2157.