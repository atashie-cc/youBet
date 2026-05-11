# Vanguard Paper Verification: International Equity Allocation Claim

**One-line verdict:** NOT VERIFIED: PDF says **35% to 55%**, not 40% to 50%. Plan's hypothesis S1 needs revision.

## Source Identified

The PDF at the cited URL is the **April 2021** revision (not the 2019 Scott/Stockton edition):
- Title: *Global equity investing: The benefits of diversification and sizing your allocation*
- Authors (2021): Scott J. Donaldson, Harshdeep Ahluwalia, Giulio Renzi-Ricci, Victor Zhu, Alexander Aleksandrovich
- Acknowledged as a revision of the 2019 paper by Brian J. Scott, Kimberly A. Stockton, and Scott J. Donaldson (which itself revised 2014 and 2008 versions)
- Data as of **September 30, 2020**

## Exact Quotes (Verbatim)

**Page 1 (Executive summary):**
> "In each market we examined, our analysis indicated that volatility was reduced most with an allocation to international equities of between **35% and 55%**. While this observation may help investors determine the appropriate mix of domestic and international equities, volatility reduction is not the only factor to consider."

**Page 4 (body text accompanying Figure 3):**
> "In each market, the marginal benefit to international diversification declines as allocations to international equities increase. The downward-curving lines in Figure 3 illustrate that volatility generally begins to rise with allocations of greater than **35% to 55%** to international equities. Similar conclusions can be drawn from a historical analysis in most of these markets; see Figure A-1 in the Appendix. The euro area is the exception; because of notably higher volatility (both historical and expected) relative to other markets, full allocation to non-euro area stocks results in the lowest expected portfolio volatility."

## Sample Period and Methodology

- **Primary analysis (Figure 3, body)**: Forward-looking VCMM (Vanguard Capital Markets Model), 10-year expected reduction in volatility, **median of 10,000 simulations as of September 30, 2020**, in local currency. Five regions: U.S., euro area, Canada, U.K., Australia.
- **Confirmatory analysis (Figure A-1, Appendix)**: Historical, with different start dates per region due to data availability — U.S. from **Jan 1, 1985**; U.K., euro area, Australia, Canada from **1999**. All ending Sept 30, 2020. MSCI indices.
- The 35-55% range refers to the **forward-looking VCMM result**, with the appendix confirming "similar conclusions" historically except for the euro area.

## Discrepancy with Plan's S1 Hypothesis

The plan cites **"40% and 50%"** (a tight 10-point range). The paper actually says **"35% and 55%"** (a 20-point range). This is a meaningfully wider band — the secondary summary tightened it.

Additional caveats the secondary summary dropped:
1. The result is forward-looking simulation (VCMM), not pure historical — historical data only "confirms similar conclusions."
2. The euro area is an explicit exception (full international allocation minimizes volatility there).
3. The range varies by domicile region; it is not a single number for U.S. investors. The U.S.-specific minimum-variance point in Figure 3a panel (US) is within this band but not the band itself.

## Recommended Revision to S1

If S1 currently operationalizes "40-50% minimizes volatility for a U.S. investor," revise to either:
- **(a) faithful to the paper:** test 35-55% as the predicted variance-minimizing region, and only for non-euro-area domiciles.
- **(b) U.S.-specific:** restrict to the U.S. panel of Figure 3a and read its actual minimum off the curve (the paper does not state a single number; it gives a curve).

The 2021 paper reused the 2019 paper's methodology but updated data to Sept 2020. The headline range of 35-55% appears in the 2021 version verbatim and is attributed to the 2019 framework. It is highly likely the original 2019 paper said the same thing (the secondary summary citing "40-50%" appears to be an error or paraphrase).
