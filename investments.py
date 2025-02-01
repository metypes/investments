if __name__ == '__main__':
    import streamlit as st
    import plotly.express as px
    import pandas as pd
    import numpy as np

    # --------------------------------------------------------
    # 1. Mock Data Definitions
    # --------------------------------------------------------

    # We'll create a dictionary of asset classes, each containing:
    #   - default_alloc: default slider value (%)
    #   - funds: dict of possible fund tickers with mock risk_factor, p_e_ratio, and monthly_returns
    # We'll generate random monthly returns for each fund to simulate performance.

    # Date range for monthly data: Jan 2021 to Dec 2023 (36 months)
    dates = pd.date_range("2021-01-01", "2023-12-31", freq="MS")
    NUM_MONTHS = len(dates)

    np.random.seed(42)  # For consistent random data


    def generate_mock_returns(risk_factor: float, size: int = 36) -> pd.Series:
        """
        Generate random monthly returns for demonstration.
        risk_factor guides the mean & volatility (roughly).
        """
        # We'll center around 0.4% monthly + small factor depending on risk_factor
        base_return = 0.004 + (risk_factor - 5) * 0.0005
        # Volatility loosely scales with risk_factor
        std_dev = 0.015 + (risk_factor - 5) * 0.002
        returns = np.random.normal(loc=base_return, scale=std_dev, size=size)
        return pd.Series(returns, index=dates)


    ASSET_CATEGORIES = {
        "Global Equities": {
            "default_alloc": 30,
            "funds": {
                "iShares MSCI World High Div. UCITS (IWDV)": {
                    "risk_factor": 7,
                    "p_e_ratio": 17,
                },
                "SPDR S&P Global Div Aristocrats (GLDV)": {
                    "risk_factor": 7,
                    "p_e_ratio": 18,
                },
                "Vanguard FTSE All-World High Div (VHYL)": {
                    "risk_factor": 7,
                    "p_e_ratio": 16,
                },
            }
        },
        "Global Bonds": {
            "default_alloc": 25,
            "funds": {
                "PIMCO GIS Global High Yield (PGHY)": {
                    "risk_factor": 5,
                    "p_e_ratio": None,  # Not typically relevant
                },
                "Amundi Emerging Markets Bond (AEMB)": {
                    "risk_factor": 6,
                    "p_e_ratio": None,
                },
            }
        },
        "Global REITs": {
            "default_alloc": 15,
            "funds": {
                "iShares Developed Markets Property Yield (IWDP)": {
                    "risk_factor": 5,
                    "p_e_ratio": 15,
                },
                "Brookfield Global REIT (BGREIT)": {
                    "risk_factor": 5,
                    "p_e_ratio": 14,
                },
            }
        },
        "Commodities & Precious Metals": {
            "default_alloc": 10,
            "funds": {
                "Amundi Commodity Index Fund (ACIF)": {
                    "risk_factor": 5,
                    "p_e_ratio": None,
                },
                "Invesco Optimum Yield Diversified Commodity (PDBC)": {
                    "risk_factor": 5,
                    "p_e_ratio": None,
                },
            }
        },
        "Rare/Strategic Metals": {
            "default_alloc": 5,
            "funds": {
                "VanEck Rare Earth/Strategic Metals UCITS (REMX)": {
                    "risk_factor": 6,
                    "p_e_ratio": None,
                },
                "Global X Lithium & Battery Tech UCITS (LIT)": {
                    "risk_factor": 7,
                    "p_e_ratio": None,
                },
            }
        },
        "Cash & Equivalents": {
            "default_alloc": 5,
            "funds": {
                "High-Interest Savings (Generic)": {
                    "risk_factor": 1,
                    "p_e_ratio": None,
                },
                "Money Market Fund (Generic)": {
                    "risk_factor": 1,
                    "p_e_ratio": None,
                },
            }
        },
    }

    # Generate monthly returns for each fund
    for category, cat_data in ASSET_CATEGORIES.items():
        for fund_name, fund_data in cat_data["funds"].items():
            rfactor = fund_data["risk_factor"]
            fund_data["monthly_returns"] = generate_mock_returns(rfactor, NUM_MONTHS)

    # --------------------------------------------------------
    # 2. Streamlit App UI
    # --------------------------------------------------------

    st.title("High-Yield Portfolio Simulator with Fund Selection")

    st.write("""
    **Demo Features**:
    - Sliders for percentage allocation per asset class (summing to ~100%).
    - Dropdowns to choose **specific funds/tickers** in each class.
    - Input your starting **portfolio balance** (default \$250k).
    - Select a backtest timeframe (1, 2, or 3 years) using **mock** data.
    - Computes overall **P/E ratio**, **profit/ROI**, **max drawdown**, and plots a cumulative return chart.

    > **All data here is fictitious, for demonstration only.**
    """)

    # Portfolio balance
    portfolio_balance = st.number_input(
        label="Portfolio Balance (USD)",
        min_value=10000,
        value=250000,
        step=5000
    )

    # Timeframe selection
    timeframe = st.selectbox(
        "Select backtest timeframe:",
        ["Last 1 Year", "Last 2 Years", "Last 3 Years"]
    )

    if timeframe == "Last 1 Year":
        months_back = 12
    elif timeframe == "Last 2 Years":
        months_back = 24
    else:
        months_back = 36

    st.write("---")

    # Container for the allocations/fund selections
    st.subheader("Asset Class Allocations & Fund Choices")

    user_inputs = {}
    total_alloc_raw = 0

    for category, cat_data in ASSET_CATEGORIES.items():
        # Slider
        alloc = st.slider(
            f"{category} Allocation (%)",
            min_value=0,
            max_value=100,
            value=cat_data["default_alloc"],
            step=5
        )
        # Dropdown for the specific fund
        possible_funds = list(cat_data["funds"].keys())
        selected_fund = st.selectbox(
            f"Select Fund/Ticker for {category}",
            options=possible_funds
        )

        user_inputs[category] = {
            "allocation": alloc,
            "fund": selected_fund
        }

        total_alloc_raw += alloc

    # Warn if not exactly 100
    st.write(f"**Raw Sum of Allocations:** {total_alloc_raw}%")
    if total_alloc_raw != 100:
        st.warning(
            "Allocations do not sum to 100% — we will normalize them."
        )

    # --------------------------------------------------------
    # 3. Normalize Allocations
    # --------------------------------------------------------
    alloc_normalized = {}
    if total_alloc_raw > 0:
        for cat, inp in user_inputs.items():
            alloc_normalized[cat] = (inp["allocation"] / total_alloc_raw) * 100
    else:
        for cat in user_inputs:
            alloc_normalized[cat] = 0

    # --------------------------------------------------------
    # 4. Compute Weighted P/E and Risk
    # --------------------------------------------------------
    weighted_pe = 0.0
    pe_weight_sum = 0.0

    weighted_risk_score = 0.0

    for category, data in user_inputs.items():
        fund_info = ASSET_CATEGORIES[category]["funds"][data["fund"]]
        risk_factor = fund_info["risk_factor"]
        p_e_ratio = fund_info["p_e_ratio"]
        weight_fraction = alloc_normalized[category] / 100.0

        # Weighted risk
        weighted_risk_score += weight_fraction * risk_factor

        # Weighted P/E
        if p_e_ratio is not None:
            # We only combine P/E where it exists
            weighted_pe += p_e_ratio * weight_fraction
            pe_weight_sum += weight_fraction

    # If no equities or p_e is None for all, result is 0
    if pe_weight_sum > 0:
        portfolio_pe = weighted_pe  # Weighted average for entire portfolio
    else:
        portfolio_pe = 0

    # Categorize risk label
    if weighted_risk_score < 3:
        risk_label = "Conservative"
    elif weighted_risk_score < 6:
        risk_label = "Moderate"
    elif weighted_risk_score < 8:
        risk_label = "Moderate-High"
    else:
        risk_label = "High"

    # --------------------------------------------------------
    # 5. Simulate Returns (last X months)
    # --------------------------------------------------------
    # Build monthly portfolio returns by summing each fund's monthly return * its weight
    returns_df = pd.DataFrame(index=dates)

    for category, data in user_inputs.items():
        fund = data["fund"]
        fund_info = ASSET_CATEGORIES[category]["funds"][fund]
        returns_df[fund] = fund_info["monthly_returns"]

    # Select last X months
    if months_back > NUM_MONTHS:
        # user wants more data than we have
        recent_returns = returns_df
    else:
        recent_returns = returns_df.iloc[-months_back:, :]

    # Calculate portfolio monthly returns
    weights = {}
    for category, norm_val in alloc_normalized.items():
        fund = user_inputs[category]["fund"]
        weights[fund] = norm_val / 100.0

    portfolio_monthly_returns = []
    for date_idx in recent_returns.index:
        row = recent_returns.loc[date_idx]
        # Weighted sum of each selected fund's return
        total_r = 0.0
        for fund_col in row.index:
            w = weights.get(fund_col, 0.0)
            total_r += row[fund_col] * w
        portfolio_monthly_returns.append(total_r)

    portfolio_monthly_returns = pd.Series(portfolio_monthly_returns, index=recent_returns.index)

    # Compute cumulative product
    cumulative_growth = (1 + portfolio_monthly_returns).cumprod()
    initial_value = portfolio_balance
    final_value = initial_value * cumulative_growth.iloc[-1]
    profit = final_value - initial_value
    roi_percent = (profit / initial_value) * 100 if initial_value else 0

    # --------------------------------------------------------
    # 6. Max Drawdown
    # --------------------------------------------------------
    running_max = cumulative_growth.cummax()
    drawdown_series = (cumulative_growth - running_max) / running_max
    max_drawdown_pct = drawdown_series.min() * 100

    # --------------------------------------------------------
    # 7. Display Results
    # --------------------------------------------------------
    st.subheader("Results")

    st.markdown("### 1. Normalized Allocations")
    alloc_table = pd.DataFrame({
        "Category": list(user_inputs.keys()),
        "Selected Fund": [user_inputs[c]["fund"] for c in user_inputs],
        "Allocation (%)": [alloc_normalized[c] for c in user_inputs]
    })
    st.dataframe(alloc_table.style.format({"Allocation (%)": "{:.1f}"}))

    st.markdown("### 2. Portfolio Metrics")
    st.write(f"**Overall P/E Ratio**: {portfolio_pe:.2f}" if portfolio_pe > 0 else "N/A (No P/E funds)")
    st.write(f"**Weighted Risk Score**: {weighted_risk_score:.1f} → {risk_label}")

    st.markdown("### 3. Simulation Performance")
    st.write(f"**Timeframe**: {timeframe}")
    st.write(f"**Initial Value**: ${initial_value:,.0f}")
    st.write(f"**Final Value**: ${final_value:,.0f}")
    st.write(f"**Profit**: ${profit:,.0f}")
    st.write(f"**ROI**: {roi_percent:.2f}%")
    st.write(f"**Max Drawdown**: {max_drawdown_pct:.2f}%")

    st.markdown("### 4. Cumulative Growth Chart")
    fig = px.line(
        x=cumulative_growth.index,
        y=cumulative_growth.values,
        labels={"x": "Date", "y": "Cumulative Growth Factor"},
        title="Portfolio Cumulative Growth (Simulation)"
    )
    st.plotly_chart(fig)

    st.markdown("""
    ---
    **Disclaimer**: This is a demonstration with *fabricated data*.  
    No real market data or performance is shown.  
    Always consult professional advice for real-world investments.
    """)
