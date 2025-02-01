if __name__ == "__main__":
    import streamlit as st
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import plotly.express as px
    from datetime import datetime, timedelta

    st.title("Multi-Asset Portfolio with GLDV, AEMB, IWDP Explicitly Included")

    st.write("""
    **Key Features**:
    - **Global Equities**: 3 sliders to pick among 20 tickers (including GLDV).
    - **Global Bonds**: 1 slider (20 tickers, including AEMB).
    - **Global REITs**: 1 slider (20 tickers, including IWDP).
    - **Commodities, Rare Metals, Cash**: 1 slider each, 20 labeled tickers each.
    - Up to **5 years** of daily data from Yahoo Finance → monthly returns.
    - Computes annualized ROI & max drawdown for 1–5 years, plus a final pie chart.

    **Disclaimer**:
    - Data from Yahoo Finance may be partial or missing, especially for new or thinly traded tickers.
    - No currency conversions, fees, or expense ratios are accounted for.
    - This is a demonstration of Streamlit + yfinance, *not* financial advice.
    """)

    # ----------------------------------------------------------------
    # 1) Labeled Ticker Lists (20 Each), Now Including GLDV, AEMB, IWDP
    # ----------------------------------------------------------------

    # Global Equities (3 Sliders) – 20 items, ensuring we include GLDV
    GLOBAL_EQUITIES_OPTIONS = [
        ("VTI",  "VTI – Vanguard Total U.S. Stock Market"),
        ("VEA",  "VEA – Vanguard FTSE Developed Markets"),
        ("VWO",  "VWO – Vanguard FTSE Emerging Markets"),
        ("IVV",  "IVV – iShares Core S&P 500"),
        ("VEU",  "VEU – Vanguard FTSE All-World ex US"),
        ("URTH", "URTH – iShares MSCI World"),
        ("SPY",  "SPY – SPDR S&P 500"),
        ("QQQ",  "QQQ – Invesco NASDAQ 100"),
        ("ACWI", "ACWI – iShares MSCI ACWI"),
        ("VWRA.L","VWRA.L – Vanguard FTSE All-World (London)"),
        ("EFA",  "EFA – iShares MSCI EAFE"),
        ("EWG",  "EWG – iShares MSCI Germany"),
        ("EWJ",  "EWJ – iShares MSCI Japan"),
        ("IEMG", "IEMG – iShares Core MSCI EM"),
        ("SCHB", "SCHB – Schwab U.S. Broad Market"),
        ("VOO",  "VOO – Vanguard S&P 500"),
        ("VXUS", "VXUS – Vanguard Total Int’l Stock"),
        ("FDVV", "FDVV – Fidelity High Dividend"),
        ("GLDV.MI", "GLDV – SPDR S&P Global Dividend Aristocrats"),  # <--- Newly Added
        ("EWU",  "EWU – iShares MSCI UK"),  # replaced a lesser used ticker
    ]

    # Global Bonds – 20 items, includes AEMB
    GLOBAL_BONDS_OPTIONS = [
        ("BND",   "BND – Vanguard Total Bond Market"),
        ("AGG",   "AGG – iShares Core US Aggregate Bond"),
        ("BNDX",  "BNDX – Vanguard Total Int’l Bond"),
        ("VWOB",  "VWOB – Vanguard EM Bond"),
        ("LQD",   "LQD – iShares iBoxx $ Invmt Grade Corp"),
        ("HYG",   "HYG – iShares iBoxx $ High Yield Corp"),
        ("TLT",   "TLT – iShares 20+ Year Treasury Bond"),
        ("SHY",   "SHY – iShares 1-3 Year Treasury Bond"),
        ("EMB",   "EMB – iShares J.P. Morgan EM Bond"),
        ("IEF",   "IEF – iShares 7-10 Year Treasury Bond"),
        ("BWX",   "BWX – SPDR Int’l Treasury Bond"),
        ("IGLB",  "IGLB – iShares Long-Term Corporate Bond"),
        ("VCIT",  "VCIT – Vanguard Intmdt-Term Corp Bond"),
        ("VCSH",  "VCSH – Vanguard Short-Term Corp Bond"),
        ("TIP",   "TIP – iShares TIPS Bond"),
        ("STIP",  "STIP – iShares 0-5 Year TIPS"),
        ("BSV",   "BSV – Vanguard Short-Term Bond"),
        ("IGOV",  "IGOV – iShares Int’l Treasury Bond"),
        ("JNK",   "JNK – SPDR Bloomberg High Yield Bond"),
        ("0P0001EDR3.F",  "AEMB – Amundi Emerging Markets Bond L Acc"),  # <--- Newly Added
    ]

    # Global REITs – 20 items, includes IWDP
    GLOBAL_REITS_OPTIONS = [
        ("VNQ",  "VNQ – Vanguard Real Estate"),
        ("REET", "REET – iShares Global REIT"),
        ("RWO",  "RWO – SPDR Dow Jones Global Real Estate"),
        ("VNQI", "VNQI – Vanguard Global ex-US Real Estate"),
        ("SCHH", "SCHH – Schwab U.S. REIT"),
        ("IYR",  "IYR – iShares U.S. Real Estate"),
        ("RWR",  "RWR – SPDR Dow Jones REIT"),
        ("DRW",  "DRW – WisdomTree Global ex-US REIT"),
        ("HAUZ", "HAUZ – Xtrackers Int’l Real Estate"),
        ("FREL", "FREL – Fidelity MSCI Real Estate"),
        ("XLRE", "XLRE – Real Estate Select Sector"),
        ("SRET", "SRET – Global X SuperDividend REIT"),
        ("PSR",  "PSR – Invesco Active U.S. Real Estate"),
        ("REM",  "REM – iShares Mortgage Real Estate"),
        ("MORT", "MORT – VanEck Mortgage REIT"),
        ("KBWY", "KBWY – Invesco KBW Premium Yield REIT"),
        ("NETL", "NETL – NETLease Corporate Real Estate"),
        ("ICF",  "ICF – iShares Cohen & Steers REIT"),
        ("USRT", "USRT – iShares Core U.S. REIT"),
        ("IWDP.MI", "IWDP – iShares Developed Mkts Property Yield"),  # <--- Newly Added
    ]

    # Commodities – 20 items
    COMMODITIES_OPTIONS = [
        ("GLD",   "GLD – SPDR Gold"),
        ("SLV",   "SLV – iShares Silver"),
        ("PDBC",  "PDBC – Invesco Optimum Yield Commodity"),
        ("DBC",   "DBC – Invesco DB Commodity Index"),
        ("BCI",   "BCI – abrdn Bloomberg All Commodity"),
        ("GSG",   "GSG – iShares S&P GSCI Commodity"),
        ("COMT",  "COMT – iShares GSCI Commodity Index"),
        ("USO",   "USO – United States Oil"),
        ("CORN",  "CORN – Teucrium Corn"),
        ("WEAT",  "WEAT – Teucrium Wheat"),
        ("DBA",   "DBA – Invesco DB Agriculture"),
        ("UGA",   "UGA – United States Gasoline"),
        ("DBB",   "DBB – Invesco DB Base Metals"),
        ("DBE",   "DBE – Invesco DB Energy"),
        ("DBP",   "DBP – Invesco DB Precious Metals"),
        ("SGOL",  "SGOL – abrdn Physical Gold"),
        ("SIVR",  "SIVR – abrdn Physical Silver"),
        ("SOYB",  "SOYB – Teucrium Soybean"),
        ("NIB",   "NIB – iPath Bloomberg Cocoa"),
        ("CANE",  "CANE – Teucrium Sugar"),
    ]

    # Rare Metals – 20 items
    RARE_METALS_OPTIONS = [
        ("REMX", "REMX – VanEck Rare Earth/Strategic Metals"),
        ("LIT",  "LIT – Global X Lithium & Battery Tech"),
        ("BATT", "BATT – Amplify Lithium & Battery Tech"),
        ("GOAU", "GOAU – U.S. Global GO GOLD & Precious Metal Miners"),
        ("PICK", "PICK – iShares MSCI Global Metals & Mining"),
        ("XME",  "XME – SPDR S&P Metals & Mining"),
        ("COPX", "COPX – Global X Copper Miners"),
        ("URA",  "URA – Global X Uranium"),
        ("KARS", "KARS – KraneShares EV & Future Mobility"),
        ("CRIT", "CRIT – NXTG Rare Earth/Critical Materials"),
        ("SLVP", "SLVP – iShares MSCI Global Silver Miners"),
        ("GDX",  "GDX – VanEck Gold Miners"),
        ("GDXJ", "GDXJ – VanEck Junior Gold Miners"),
        ("PLTM", "PLTM – GraniteShares Platinum Trust"),
        ("PALL", "PALL – abrdn Physical Palladium"),
        ("SIL",  "SIL – Global X Silver Miners"),
        ("GLTR", "GLTR – abrdn Physical Precious Metals"),
        ("BAR",  "BAR – abrdn Physical Gold Shares"),
        ("SGDM", "SGDM – Sprott Gold Miners"),
        ("SILJ", "SILJ – ETFMG Prime Junior Silver"),
    ]

    # Cash & Equivalents – 20 items
    CASH_EQUIVALENTS_OPTIONS = [
        ("BIL",   "BIL – SPDR Bloomberg 1-3 Month T-Bill"),
        ("SHV",   "SHV – iShares Short Treasury Bond"),
        ("ICSH",  "ICSH – iShares Ultra Short-Term Bond"),
        ("NEAR",  "NEAR – iShares Short Maturity Bond"),
        ("MINT",  "MINT – PIMCO Enhanced Short Maturity"),
        ("GSY",   "GSY – Invesco Ultra Short Duration"),
        ("GBIL",  "GBIL – Goldman Sachs Access Treasury 0-1 Year"),
        ("CLTL",  "CLTL – Invesco Treasury Collateral"),
        ("JPST",  "JPST – JPMorgan Ultra-Short Income"),
        ("SCHO",  "SCHO – Schwab Short-Term U.S. Treasury"),
        ("MEAR",  "MEAR – iShares Short Maturity Municipal"),
        ("FLOT",  "FLOT – iShares Floating Rate Bond"),
        ("TBIL",  "TBIL – U.S. 3 Month Treasury Bill ETF"),
        ("BILS",  "BILS – SPDR Bloomberg 3-12 Month T-Bill"),
        ("LDUR",  "LDUR – PIMCO Low Duration"),
        ("ULST",  "ULST – SPDR SSGA Ultra Short Term Bond"),
        ("VRIG",  "VRIG – Invesco Variable Rate Investment Grade"),
        ("SUB",   "SUB – iShares Short-Term National Muni Bond"),
        ("VUSB",  "VUSB – Vanguard Ultra-Short Bond"),
        ("FLDR",  "FLDR – Fidelity Low Duration Bond Factor"),
    ]

    def get_symbol_from_label(chosen_label, options):
        """Given 'TICKER – Full Name', return 'TICKER' from the list of tuples."""
        for sym, lbl in options:
            if lbl == chosen_label:
                return sym
        return None

    # ----------------------------------------------------------------
    # 2) Set up Date Range for up to 5 Years
    # ----------------------------------------------------------------
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 5)
    st.write(f"**Data range**: {start_date.date()} → {end_date.date()} (up to 5 years)")

    # ----------------------------------------------------------------
    # 3) Global Equities – 3 Sliders
    # ----------------------------------------------------------------
    st.subheader("Global Equities (3 Sliders)")

    c1, c2, c3 = st.columns(3)
    with c1:
        eq_label1 = st.selectbox("Equity Option 1", [x[1] for x in GLOBAL_EQUITIES_OPTIONS], index=0)
        eq_alloc1 = st.slider("Alloc (%) E1", 0, 100, 10, 5)
    with c2:
        eq_label2 = st.selectbox("Equity Option 2", [x[1] for x in GLOBAL_EQUITIES_OPTIONS], index=1)
        eq_alloc2 = st.slider("Alloc (%) E2", 0, 100, 10, 5)
    with c3:
        eq_label3 = st.selectbox("Equity Option 3", [x[1] for x in GLOBAL_EQUITIES_OPTIONS], index=2)
        eq_alloc3 = st.slider("Alloc (%) E3", 0, 100, 10, 5)

    eq_ticker1 = get_symbol_from_label(eq_label1, GLOBAL_EQUITIES_OPTIONS)
    eq_ticker2 = get_symbol_from_label(eq_label2, GLOBAL_EQUITIES_OPTIONS)
    eq_ticker3 = get_symbol_from_label(eq_label3, GLOBAL_EQUITIES_OPTIONS)

    total_equity_alloc = eq_alloc1 + eq_alloc2 + eq_alloc3
    st.write(f"**Sum of these 3 Equity allocations**: {total_equity_alloc}%")

    # ----------------------------------------------------------------
    # 4) Other Asset Classes
    # ----------------------------------------------------------------
    def pick_asset(label, option_list, default_alloc):
        lab = st.selectbox(f"{label} Ticker", [x[1] for x in option_list], index=0)
        al = st.slider(f"{label} Allocation (%)", 0, 100, default_alloc, step=5)
        sym = get_symbol_from_label(lab, option_list)
        return sym, lab, al

    st.subheader("Other Asset Classes (1 Slider Each)")

    bond_ticker, bond_label, bond_alloc = pick_asset("Global Bonds", GLOBAL_BONDS_OPTIONS, 20)
    reit_ticker, reit_label, reit_alloc = pick_asset("Global REITs", GLOBAL_REITS_OPTIONS, 10)
    comm_ticker, comm_label, comm_alloc = pick_asset("Commodities", COMMODITIES_OPTIONS, 10)
    rare_ticker, rare_label, rare_alloc = pick_asset("Rare Metals", RARE_METALS_OPTIONS, 5)
    cash_ticker, cash_label, cash_alloc = pick_asset("Cash & Equivalents", CASH_EQUIVALENTS_OPTIONS, 5)

    raw_total_alloc = total_equity_alloc + bond_alloc + reit_alloc + comm_alloc + rare_alloc + cash_alloc
    st.write(f"**Raw total allocation**: {raw_total_alloc}%")
    if raw_total_alloc != 100:
        st.warning("Allocations do not sum to 100% – will normalize automatically.")

    # ----------------------------------------------------------------
    # 5) Gather Tickers & Download Data
    # ----------------------------------------------------------------
    chosen_tickers = [
        eq_ticker1, eq_ticker2, eq_ticker3,
        bond_ticker, reit_ticker, comm_ticker,
        rare_ticker, cash_ticker
    ]
    chosen_tickers = list({x for x in chosen_tickers if x})  # remove duplicates/None

    if len(chosen_tickers) == 0:
        st.error("No valid tickers selected. Stopping.")
        st.stop()

    st.write(f"**Fetching data** for: {', '.join(chosen_tickers)}")
    raw_data = yf.download(chosen_tickers, start=start_date, end=end_date, progress=False)['Close']


    if raw_data.empty:
        st.error("No data returned from Yahoo Finance. Check tickers or date range.")
        st.stop()

    # if len(chosen_tickers) > 1:
    #     # Multi-index => we can do
    #     print(raw_data.columns)
    #     raw_data = raw_data["Adj Close"]  # a DataFrame of shape (dates, tickers)
    # else:
    #     # Single ticker => 'data' is single-level.
    #     # 'data["Adj Close"]' is a Series of shape (dates,)
    #     raw_data = raw_data[["Adj Close"]]  # Make it a DataFrame
    #     raw_data.columns = [chosen_tickers[0]]

    # Ensure DataFrame shape if only 1 ticker
    if len(chosen_tickers) == 1:
        raw_data = pd.DataFrame(raw_data)

    # Resample daily => monthly
    monthly_prices = raw_data.resample("M").last()
    monthly_prices.dropna(how="all", axis=1, inplace=True)
    monthly_returns = monthly_prices.pct_change().dropna(how="all")

    # ----------------------------------------------------------------
    # 6) Weighted Portfolio
    # ----------------------------------------------------------------
    if raw_total_alloc == 0:
        st.error("Total allocation is 0%. Nothing to compute.")
        st.stop()

    eq_w1 = eq_alloc1 / raw_total_alloc
    eq_w2 = eq_alloc2 / raw_total_alloc
    eq_w3 = eq_alloc3 / raw_total_alloc
    bond_w = bond_alloc / raw_total_alloc
    reit_w = reit_alloc / raw_total_alloc
    comm_w = comm_alloc / raw_total_alloc
    rare_w = rare_alloc / raw_total_alloc
    cash_w = cash_alloc / raw_total_alloc

    weight_map = {
        eq_ticker1: eq_w1,
        eq_ticker2: eq_w2,
        eq_ticker3: eq_w3,
        bond_ticker: bond_w,
        reit_ticker: reit_w,
        comm_ticker: comm_w,
        rare_ticker: rare_w,
        cash_ticker: cash_w,
    }

    valid_cols = [c for c in weight_map if c in monthly_returns.columns]
    portfolio_df = monthly_returns[valid_cols].copy()
    for col in portfolio_df.columns:
        portfolio_df[col] *= weight_map[col]

    portfolio_monthly_returns = portfolio_df.sum(axis=1)
    portfolio_cumulative = (1 + portfolio_monthly_returns).cumprod()

    # ----------------------------------------------------------------
    # 7) Annualized Stats (1–5 Years)
    # ----------------------------------------------------------------
    def annualized_return_factor(cum_series):
        if len(cum_series) < 2:
            return 0
        total_m = len(cum_series)
        final_val = cum_series.iloc[-1]
        return (final_val ** (12 / total_m)) - 1

    def max_drawdown(cum_series):
        peak = cum_series.cummax()
        dd = (cum_series - peak) / peak
        return dd.min()

    def stats_for_period(months):
        if months > len(portfolio_cumulative):
            sub = portfolio_cumulative
        else:
            sub = portfolio_cumulative.iloc[-months:]
        if len(sub) < 2:
            return 0, 0
        ann_ret = annualized_return_factor(sub)
        dd_ = max_drawdown(sub)
        return ann_ret, dd_

    horizon_map = {"1Y": 12, "2Y": 24, "3Y": 36, "4Y": 48, "5Y": 60}
    rows = []
    for label, mcount in horizon_map.items():
        ret_, dd_ = stats_for_period(mcount)
        rows.append({
            "Horizon": label,
            "Annualized ROI (%)": f"{ret_*100:.2f}",
            "Max Drawdown (%)": f"{dd_*100:.2f}",
        })
    stats_df = pd.DataFrame(rows)

    st.subheader("Annualized Returns & Drawdown (1–5 Years)")
    st.dataframe(stats_df)

    # ----------------------------------------------------------------
    # 8) Pie Chart
    # ----------------------------------------------------------------
    eq_lbl1 = f"Equity 1: {eq_label1}" if eq_label1 else "Equity 1"
    eq_lbl2 = f"Equity 2: {eq_label2}" if eq_label2 else "Equity 2"
    eq_lbl3 = f"Equity 3: {eq_label3}" if eq_label3 else "Equity 3"

    pie_data = [
        {"Label": eq_lbl1,      "Alloc %": eq_alloc1},
        {"Label": eq_lbl2,      "Alloc %": eq_alloc2},
        {"Label": eq_lbl3,      "Alloc %": eq_alloc3},
        {"Label": f"Bonds: {bond_label}",         "Alloc %": bond_alloc},
        {"Label": f"REITs: {reit_label}",         "Alloc %": reit_alloc},
        {"Label": f"Commodities: {comm_label}",   "Alloc %": comm_alloc},
        {"Label": f"Rare Metals: {rare_label}",   "Alloc %": rare_alloc},
        {"Label": f"Cash: {cash_label}",          "Alloc %": cash_alloc},
    ]
    pie_df = pd.DataFrame(pie_data)

    st.subheader("Final Allocation Pie Chart")
    fig_pie = px.pie(
        pie_df,
        names="Label",
        values="Alloc %",
        title="Portfolio Allocation Breakdown",
        hover_data=["Alloc %"]
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie)

    # ----------------------------------------------------------------
    # 9) Full-Period Cumulative Return
    # ----------------------------------------------------------------
    st.subheader("Full-Period Cumulative Return")
    if len(portfolio_cumulative) < 2:
        st.info("Not enough data to plot a cumulative chart.")
    else:
        fig_line = px.line(
            x=portfolio_cumulative.index,
            y=portfolio_cumulative.values,
            labels={"x": "Date", "y": "Growth Factor"},
            title="Portfolio Cumulative Return (Entire Available Period)"
        )
        st.plotly_chart(fig_line)

        start_str = portfolio_cumulative.index[0].strftime("%Y-%m-%d")
        end_str = portfolio_cumulative.index[-1].strftime("%Y-%m-%d")
        final_factor = portfolio_cumulative.iloc[-1]
        total_return = final_factor - 1
        dd_full = max_drawdown(portfolio_cumulative) * 100
        st.write(f"**Total Return** from {start_str} to {end_str}: {total_return*100:.2f}%")
        st.write(f"**Max Drawdown**: {dd_full:.2f}%")

    st.markdown("""
    ---
    **Note**:
    - We've explicitly included **GLDV**, **AEMB**, and **IWDP** in the Global Equities, Bonds, and REITs lists, respectively.
    - Some tickers may have incomplete or short history in Yahoo Finance.
    - No currency conversions or fee assumptions are made. 
    - This is a **demonstration** of multi-asset portfolio simulation with labeled tickers, *not* financial advice.
    """)
