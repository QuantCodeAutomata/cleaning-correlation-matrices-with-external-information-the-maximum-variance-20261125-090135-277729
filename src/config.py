"""Parameters for both experiments, taken from Bun, Bouchaud & Potters,
"Cleaning correlation matrices with external information: the Maximum
Variance estimator" (sections referenced below). Every value lists its
meaning, unit and source.

The 47 futures contracts of Table 2 (24 stock indices, 14 government
bonds, 9 currencies) are mostly listed on exchanges (Eurex, ICE, SGX, TSE,
ASX, KRX, NSE, Montreal Exchange) that the project's market-data provider
(QuantX Dataloader / Massive) does not serve: direct index symbols
(I:SPX, I:NDX, I:DJI) returned NOT_AUTHORIZED, and the Massive futures
API covers only CME/CBOT/NYMEX/COMEX-cleared products. UNIVERSE below
therefore maps each Table 2 instrument to the closest long-history,
US-accessible proxy (a country/region equity ETF or closed-end fund, a
duration-matched US or international government-bond ETF, or a spot
currency pair), grouped into the same three asset classes the paper uses
as the K=3 MaxVar groups. This substitution is recorded in
results/RESULTS.md.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Universe: Table 2 instrument -> (proxy ticker, asset-class group)
# Groups follow the paper's K=3 partition: stock indices, bonds, currencies.
# ---------------------------------------------------------------------------

STOCK_INDEX_UNIVERSE = {
    # paper instrument (Table 2): proxy ticker used for this replication
    "AEX (Netherlands)": "EWN",
    "CAC 40 (France)": "EWQ",
    "DAX (Germany)": "EWG",
    "Dow Jones Industrial Average (US)": "DIA",
    "Euro Stoxx 50": "EZU",
    "FTSE 100 (UK)": "EWU",
    "FTSE MIB (Italy)": "EWI",
    "Hang Seng (Hong Kong)": "EWH",
    "IBEX 35 (Spain)": "EWP",
    "Kospi 200 (Korea)": "EWY",
    "MSCI Emerging Markets": "EEM",
    "MSCI EAFE": "EFA",
    "MSCI Singapore": "EWS",
    "MSCI Taiwan": "EWT",
    "Nasdaq 100 (US)": "QQQ",
    "Nifty (India)": "INDA",
    "Nikkei 225 (Japan)": "EWJ",
    "S&P/TSX 60 (Canada)": "EWC",
    "S&P 500 (US)": "SPY",
    "S&P MidCap 400 (US)": "MDY",
    "S&P/ASX 200 (Australia)": "EWA",
    "STOXX Europe 600": "IEV",
    "Topix (Japan)": "DXJ",
    "Russell 2000 (US)": "IWM",
}

# The paper's 14 government-bond futures span five countries (US, Germany,
# France, Italy, Canada, Japan) at several maturities. Long-history,
# single-country sovereign-bond ETFs are not available for Germany, France,
# Italy, Canada or Japan individually through this project's data sources;
# using the same broad "international treasury" ETF for several Table 2
# instruments would make those columns of C numerically identical (perfect
# collinearity), which is not a measurement-noise problem the estimators are
# designed to clean. The bond group below therefore keeps only the US
# maturities (distinct proxies) plus the distinct non-US treasury ETFs that
# are available, instead of repeating a proxy across several Table 2 rows;
# the resulting universe has 42 (not 47) unique instruments (see
# UNIVERSE_SIZE_RUN below and results/RESULTS.md).
BOND_UNIVERSE = {
    "US Treasury note 2Y": "SHY",
    "US Treasury note 5Y": "IEI",
    "US Treasury note 10Y": "IEF",
    "US Treasury bond 30Y": "TLT",
    "US Ultra Treasury bond": "EDV",
    "International Treasury (ex-US, Bund/OAT/BTP/JGB proxy)": "IGOV",
    "International Treasury Bond, short-duration (ex-US)": "ISHG",
    "International Government Bond (ex-US)": "BWX",
    "Global Government Bond ex-US": "BNDX",
}

CURRENCY_UNIVERSE = {
    "EUR": "C:EURUSD",
    "JPY": "C:USDJPY",
    "GBP": "C:GBPUSD",
    "CHF": "C:USDCHF",
    "CAD": "C:USDCAD",
    "AUD": "C:AUDUSD",
    "NZD": "C:NZDUSD",
    "MXN": "C:USDMXN",
    "US Dollar Index": "UUP",
}

# Pairs quoted as USD per 1 foreign-currency unit must be inverted to the
# common "USD value of one foreign-currency unit" convention used for all
# currencies in the universe.
CURRENCY_INVERT = {"C:USDJPY", "C:USDCHF", "C:USDCAD", "C:USDMXN"}

GROUPS = {
    "stock_indices": STOCK_INDEX_UNIVERSE,
    "bonds": BOND_UNIVERSE,
    "currencies": CURRENCY_UNIVERSE,
}
N_GROUPS = len(GROUPS)  # K = 3 (paper, Section 3: the three asset classes)

# ---------------------------------------------------------------------------
# Sample period (paper, Section 3 / Table 2 caption)
# ---------------------------------------------------------------------------
SAMPLE_START_PAPER = "1985-05-08"  # paper: start of daily price history
SAMPLE_END_PAPER = "2018-12-31"  # paper: end of daily price history
REFERENCE_DATE = "2018-12-28"  # paper: date of the MaxVar reference matrix C0
OOS_START_PAPER = "1993-01-01"  # paper: start of the out-of-sample Sharpe-ratio window
OOS_END_PAPER = "2018-12-31"  # paper: end of the out-of-sample Sharpe-ratio window

# ---------------------------------------------------------------------------
# Exponential moving average parameters (paper, Eqs. 10-11 and Section 3)
# ---------------------------------------------------------------------------
ETA_VOL = 1.0 / 100.0  # paper Eq. 10: EMA parameter for daily volatility (eta), applied to daily returns

# Paper, Section 3 (between Eqs. 11 and 12): "With eta' = 1/750 in units of
# days, the estimation window contains about 150 independent weekly returns
# for 47 contracts." The correlation EMA of Eq. 11 is a recursion over
# *weekly* returns, so the day-unit label eta'=1/750 is not applied literally
# to that weekly recursion; it is converted to a weekly decay parameter by
# the number of trading days per week (5 trading days/week): 750 days / 5 =
# 150 weeks, i.e. eta'_weekly = 5/750 = 1/150. Likewise eta'=1/100 in day
# units (Table 3, "shortened to eta'=1/100, i.e. about 20 weekly returns")
# converts to eta'_weekly = 5/100 = 1/20.
TRADING_DAYS_PER_WEEK = 5  # unit conversion, paper Section 3 (day-unit eta' -> weekly eta')
ETA_CORR_SLOW_DAYUNITS = 1.0 / 750.0  # paper Section 3 / Table 1: eta' quoted in units of days
ETA_CORR_FAST_DAYUNITS = 1.0 / 100.0  # paper Section 3 / Table 3: eta' quoted in units of days
ETA_CORR_SLOW = TRADING_DAYS_PER_WEEK * ETA_CORR_SLOW_DAYUNITS  # = 1/150: weekly decay for Eq. 11
ETA_CORR_FAST = TRADING_DAYS_PER_WEEK * ETA_CORR_FAST_DAYUNITS  # = 1/20: weekly decay for Eq. 11

# ---------------------------------------------------------------------------
# Experiment 1: Monte Carlo noise-reduction test (paper Section 3, Table 1)
# ---------------------------------------------------------------------------
MC_N_SAMPLES = 100  # paper: number of independent Monte Carlo samples
MC_T_PAPER = 150  # paper: length T of each simulated Gaussian sample
MC_SEED = 0  # not specified in the paper; fixed here for reproducibility

EXPECTED_FROBENIUS_MAXVAR = 2.09  # paper, Table 1
EXPECTED_FROBENIUS_RIE = 2.99  # paper, Table 1
EXPECTED_FROBENIUS_EMPIRICAL = 3.24  # paper, Table 1

# ---------------------------------------------------------------------------
# Experiment 2: risk-parity Sharpe ratio (paper Section 3, Table 3)
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252  # annualization constant, not specified in the paper
WEEKS_PER_YEAR = 52  # annualization constant, not specified in the paper

# e has components 1 for stock indices and bonds, 0 for currencies
# (paper Section 3: the risk-parity portfolio hedges equities and bonds only).
RISK_PARITY_INCLUDE_GROUPS = ("stock_indices", "bonds")

EXPECTED_SHARPE_ETA_SLOW = {"empirical": 1.17, "rie": 1.24, "maxvar": 1.28}  # Table 3
EXPECTED_SHARPE_ETA_FAST = {"empirical": None, "rie": 0.93, "maxvar": 1.05}  # Table 3

# ---------------------------------------------------------------------------
# Universe size. The paper's Table 2 lists 47 futures contracts. Deduplicating
# the bond-group proxies (see BOND_UNIVERSE above) leaves 42 distinct
# instruments actually loaded and used in both experiments; the deduplication
# reason and its effect are recorded in results/RESULTS.md. Downloading this
# 42-instrument universe over the full 1985-2018 window takes a few minutes,
# well under the time available, so no further reduction for download time is
# applied.
# ---------------------------------------------------------------------------
UNIVERSE_SIZE_PAPER = 47
UNIVERSE_SIZE_RUN = 42
