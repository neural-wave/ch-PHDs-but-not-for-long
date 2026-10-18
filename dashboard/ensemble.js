// "Quantile ensemble" tab: static report of the research champion on branch
// el_bicho/hackaton-research of ../ch-PHDs-but-not-for-long/
// (minimize_expanding_20261017/fixed_blend_ratio), its RevIN refit and its
// frozen deployment refit. The figures below are transcribed from that repo's
// results/minimize_expanding_20261017/, results/champion_revin_20261017/ and
// results/illustrative_quantile_bands_20261017/, not computed here, so they
// don't depend on the dashboard filters - update them when a campaign is
// rerun. Only the example days' observed prices come from DATASET. Classic
// script - exposes a single global `Ensemble`.

const Ensemble = (function () {
  const ZONE_COLOR_VAR = {
    ZONE_0: "--series-zone0",
    ZONE_1: "--series-zone1",
    ZONE_2: "--series-zone2",
    ZONE_3: "--series-zone3",
  };
  const FONT = "11px ui-monospace, 'SF Mono', Menlo, Consolas, monospace";

  const MONTHS = {
    labels: ["Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"],
    trainMonths: [3, 4, 5, 6, 7, 8, 9, 10],
  };

  // monthly_scores.csv - q50 MAE (EUR/MWh) of fixed_blend_ratio and of
  // mean-corrected linear interpolation, each test month, pooled over zones.
  const MONTHLY_MAE = {
    ensemble: [2.903, 4.484, 4.377, 3.974, 3.862, 3.723, 4.340, 5.091],
    interpolation: [3.849, 6.521, 5.832, 5.646, 5.839, 5.380, 6.238, 7.090],
  };

  // monthly_zone_scores.csv - same-zone ratio MAE / interpolation MAE.
  const ZONE_RATIO = {
    ZONE_0: [0.626, 0.551, 0.622, 0.584, 0.530, 0.580, 0.599, 0.644],
    ZONE_1: [0.674, 0.621, 0.691, 0.637, 0.581, 0.624, 0.654, 0.735],
    ZONE_2: [0.775, 0.843, 0.877, 0.811, 0.826, 0.822, 0.846, 0.781],
    ZONE_3: [0.874, 0.827, 0.846, 0.824, 0.770, 0.747, 0.684, 0.694],
  };

  // champion_revin_20261017/monthly_scores.csv - monthly ratio of the original
  // champion and of the same frozen recipe freshly fitted with RevIN.
  const REVIN_RATIO = {
    original: [0.7543, 0.6877, 0.7506, 0.7039, 0.6615, 0.6920, 0.6958, 0.7180],
    revin: [0.7586, 0.7000, 0.7829, 0.7303, 0.6733, 0.7089, 0.7095, 0.7376],
  };

  // champion_revin_20261017/example_predictions.csv - the two example days, 96
  // quarters each: calibrated 90% bounds and raw q25/q50/q75, EUR/MWh. Panel
  // stats from examples.json. Observed prices are read from DATASET.
  const EXAMPLES = [
    {
      key: "ex1", date: "2026-03-14", zone: "ZONE_2",
      original: {
        width90: 7.04, mae: 3.28, coverage90: 64.6,
        lo90: [145.40, 144.13, 140.75, 136.95, 141.42, 139.32, 137.25, 133.98, 135.87, 134.29, 132.99, 132.10, 137.00, 138.43, 137.75, 135.01, 132.65, 131.73, 131.15, 129.39, 126.26, 127.08, 129.67, 133.22, 135.31, 144.97, 149.00, 150.22, 148.85, 150.96, 150.06, 146.75, 145.19, 142.90, 140.41, 136.16, 141.28, 138.37, 135.83, 132.70, 142.33, 140.45, 138.37, 135.41, 141.29, 139.77, 137.03, 131.27, 127.24, 122.95, 120.34, 117.85, 118.33, 118.40, 118.55, 118.58, 116.76, 119.66, 122.16, 124.68, 115.56, 121.28, 127.72, 135.47, 124.06, 137.57, 149.93, 159.27, 140.55, 151.17, 159.51, 167.14, 155.44, 164.62, 173.92, 178.83, 181.75, 181.73, 174.46, 163.49, 165.67, 156.78, 148.65, 140.07, 149.03, 144.62, 138.44, 130.52, 137.32, 134.81, 132.81, 129.51, 137.00, 134.70, 131.84, 126.81],
        q25: [147.36, 145.71, 142.62, 139.40, 143.21, 140.78, 138.86, 136.27, 137.57, 135.67, 134.61, 134.26, 138.93, 139.91, 139.34, 137.36, 134.70, 133.37, 132.79, 131.71, 128.49, 128.76, 131.39, 135.70, 138.01, 146.90, 150.94, 152.92, 151.15, 152.73, 151.85, 149.17, 147.27, 144.46, 142.11, 138.48, 143.20, 139.90, 137.57, 135.08, 144.09, 141.87, 140.04, 137.62, 143.13, 141.24, 138.52, 133.28, 128.90, 124.21, 121.77, 119.64, 120.06, 119.51, 119.94, 120.37, 118.85, 121.03, 123.78, 126.53, 118.17, 122.93, 129.61, 137.87, 128.45, 140.67, 152.49, 162.70, 144.75, 154.20, 162.17, 170.74, 159.50, 167.86, 176.49, 181.92, 185.10, 184.57, 177.62, 167.42, 168.57, 158.98, 150.98, 143.35, 151.75, 146.76, 140.52, 133.54, 139.95, 136.88, 134.79, 132.05, 139.27, 136.50, 133.88, 129.72],
        q50: [148.58, 146.60, 143.71, 140.65, 144.28, 141.59, 139.80, 137.39, 138.57, 136.43, 135.49, 135.29, 140.10, 140.71, 140.28, 138.42, 135.84, 134.24, 133.74, 132.80, 129.74, 129.77, 132.39, 136.89, 140.07, 148.05, 152.19, 154.35, 152.72, 153.72, 153.12, 150.41, 148.52, 145.38, 143.19, 139.65, 144.33, 140.74, 138.64, 136.29, 145.23, 142.74, 141.11, 138.71, 144.22, 142.03, 139.48, 134.27, 129.79, 124.93, 122.57, 120.49, 121.09, 120.25, 120.77, 121.25, 120.03, 121.90, 124.67, 127.47, 119.63, 124.09, 130.63, 139.09, 131.26, 142.83, 154.05, 164.59, 147.34, 156.36, 163.71, 172.59, 162.10, 169.88, 177.91, 183.72, 187.44, 186.06, 179.49, 169.72, 170.27, 160.27, 152.66, 145.11, 153.34, 147.91, 141.95, 135.20, 141.39, 137.90, 135.94, 133.46, 140.63, 137.44, 135.07, 131.21],
        q75: [149.99, 147.57, 144.61, 141.87, 145.52, 142.43, 140.56, 138.51, 139.67, 137.19, 136.22, 136.36, 141.18, 141.52, 141.13, 139.62, 136.94, 135.07, 134.58, 134.05, 130.86, 130.69, 133.29, 138.43, 141.52, 149.21, 153.68, 156.21, 154.25, 154.72, 154.15, 151.93, 149.89, 146.27, 144.12, 140.89, 145.83, 141.62, 139.47, 137.41, 146.65, 143.49, 141.89, 139.83, 145.45, 142.69, 140.24, 135.36, 130.80, 125.49, 123.18, 121.47, 122.05, 120.84, 121.45, 122.37, 121.01, 122.53, 125.52, 128.80, 120.70, 124.88, 131.68, 141.02, 132.95, 144.02, 155.87, 168.20, 148.89, 157.89, 165.49, 175.97, 163.75, 171.26, 179.57, 187.05, 189.99, 187.81, 181.07, 171.92, 172.30, 161.65, 153.83, 146.79, 155.15, 149.19, 143.09, 136.76, 143.00, 138.92, 136.94, 134.90, 142.22, 138.59, 136.05, 132.56],
        hi90: [152.53, 149.67, 146.27, 143.78, 147.81, 144.16, 142.03, 140.28, 141.88, 138.84, 137.57, 138.08, 143.33, 143.35, 142.72, 141.61, 139.21, 136.84, 136.31, 136.09, 133.40, 132.57, 135.04, 140.74, 144.24, 151.50, 155.88, 159.04, 157.05, 156.58, 155.83, 154.25, 152.72, 147.97, 145.70, 142.61, 148.59, 143.25, 140.85, 139.13, 149.27, 144.94, 143.21, 141.47, 147.73, 144.01, 141.58, 137.09, 132.85, 126.66, 124.37, 123.01, 123.92, 122.18, 122.57, 124.11, 122.87, 123.95, 126.93, 130.86, 122.79, 126.71, 133.60, 144.02, 136.00, 146.64, 158.98, 173.99, 152.03, 160.87, 168.94, 180.85, 167.24, 173.97, 183.00, 191.61, 194.62, 191.40, 183.85, 175.19, 176.07, 164.24, 156.09, 149.39, 158.55, 151.60, 145.32, 139.21, 145.87, 141.04, 139.22, 137.30, 145.21, 140.90, 138.07, 134.72],
      },
      revin: {
        width90: 9.91, mae: 3.17, coverage90: 82.3,
        lo90: [143.85, 143.62, 139.86, 134.50, 140.15, 138.96, 136.62, 132.36, 134.32, 133.84, 132.37, 130.82, 135.61, 137.98, 137.03, 133.08, 131.52, 131.57, 130.48, 127.59, 124.34, 126.66, 128.99, 131.57, 132.52, 143.60, 147.96, 148.09, 146.04, 149.80, 149.12, 144.34, 143.36, 142.42, 139.17, 133.63, 139.15, 137.59, 135.03, 130.21, 140.15, 139.49, 137.29, 132.28, 139.60, 139.27, 136.13, 128.35, 126.26, 122.85, 119.59, 116.11, 116.97, 118.18, 117.54, 116.43, 115.72, 119.52, 121.27, 122.15, 114.76, 121.10, 126.55, 132.62, 123.71, 137.02, 148.66, 155.84, 138.14, 149.59, 158.73, 164.87, 153.05, 163.62, 172.86, 176.17, 179.13, 180.73, 173.11, 160.20, 163.59, 156.40, 148.27, 138.82, 147.20, 144.01, 137.91, 129.21, 136.15, 134.52, 132.07, 127.91, 135.35, 134.12, 131.03, 124.75],
        q25: [146.93, 145.82, 142.47, 138.66, 142.89, 140.93, 138.75, 135.84, 137.08, 135.76, 134.52, 134.03, 138.71, 140.06, 139.21, 136.70, 134.71, 133.64, 132.64, 130.97, 128.16, 128.87, 131.20, 135.21, 137.45, 146.60, 150.55, 152.41, 150.05, 152.58, 151.71, 148.65, 146.94, 144.73, 141.58, 137.61, 142.47, 139.92, 137.51, 134.32, 143.33, 141.87, 139.81, 136.56, 142.74, 141.49, 138.34, 131.97, 128.90, 124.56, 121.47, 118.85, 119.97, 119.92, 119.54, 119.38, 118.98, 121.55, 123.53, 125.12, 118.80, 123.45, 128.99, 136.10, 129.48, 141.17, 151.67, 160.24, 143.92, 153.62, 161.86, 169.85, 159.06, 167.77, 175.78, 180.53, 183.47, 184.28, 177.27, 166.76, 167.23, 158.88, 151.26, 143.42, 150.84, 146.57, 140.63, 133.73, 139.82, 137.10, 134.65, 131.53, 138.66, 136.51, 133.81, 129.27],
        q50: [148.40, 146.77, 143.82, 140.55, 144.09, 141.74, 139.80, 137.44, 138.22, 136.54, 135.50, 135.52, 140.04, 140.92, 140.24, 138.31, 135.97, 134.47, 133.62, 132.56, 129.59, 129.81, 132.28, 137.12, 139.82, 147.95, 152.16, 154.74, 152.13, 153.83, 153.23, 150.76, 148.62, 145.77, 142.90, 139.47, 144.06, 140.93, 138.75, 136.26, 145.05, 142.94, 141.24, 138.57, 144.32, 142.42, 139.50, 133.76, 130.01, 125.27, 122.34, 120.16, 121.25, 120.67, 120.50, 120.92, 120.36, 122.45, 124.56, 126.70, 120.52, 124.59, 130.13, 138.20, 132.61, 143.32, 153.48, 163.32, 147.07, 155.94, 163.70, 173.29, 162.33, 170.11, 177.63, 183.55, 186.69, 186.10, 179.86, 170.06, 169.34, 160.11, 152.96, 145.90, 152.59, 147.67, 142.12, 136.01, 141.32, 138.10, 135.87, 133.40, 140.19, 137.51, 135.22, 131.43],
        q75: [150.45, 148.17, 144.93, 141.97, 145.79, 142.85, 140.73, 138.63, 139.79, 137.57, 136.37, 136.67, 141.60, 142.01, 141.22, 139.65, 137.51, 135.52, 134.58, 133.86, 131.31, 131.06, 133.39, 138.67, 142.08, 149.63, 153.91, 156.98, 154.44, 155.30, 154.70, 152.70, 150.73, 147.17, 144.14, 141.03, 146.24, 142.36, 139.94, 137.72, 147.36, 144.34, 142.46, 140.13, 146.31, 143.63, 140.57, 135.17, 131.51, 126.20, 123.13, 121.24, 122.84, 121.69, 121.40, 122.18, 121.93, 123.48, 125.56, 128.15, 122.21, 125.81, 131.46, 140.16, 135.07, 145.05, 155.77, 166.95, 149.62, 158.06, 165.93, 177.01, 165.10, 172.08, 179.89, 187.19, 190.28, 188.66, 182.15, 172.92, 172.05, 161.74, 154.49, 147.75, 154.98, 149.23, 143.52, 137.61, 143.39, 139.42, 137.14, 134.82, 142.46, 139.04, 136.52, 132.88],
        hi90: [154.50, 151.32, 147.49, 145.14, 149.13, 145.35, 142.90, 141.39, 143.00, 139.91, 138.41, 139.45, 144.89, 144.59, 143.59, 142.86, 140.75, 137.87, 136.87, 137.07, 134.77, 133.69, 135.84, 142.35, 145.97, 152.89, 157.71, 161.84, 158.46, 158.51, 157.76, 156.69, 154.56, 149.96, 147.02, 143.99, 150.06, 145.27, 142.72, 140.71, 151.37, 147.22, 145.30, 143.04, 149.81, 146.17, 143.13, 138.09, 134.32, 128.19, 125.10, 123.70, 125.79, 124.01, 123.37, 125.16, 124.94, 125.85, 127.87, 131.50, 125.33, 128.54, 134.35, 144.54, 138.94, 148.41, 160.18, 174.43, 153.98, 162.03, 170.23, 184.05, 169.74, 175.84, 184.53, 193.30, 196.64, 194.37, 186.50, 177.15, 176.62, 165.26, 157.53, 151.26, 159.78, 152.58, 146.59, 141.15, 147.46, 142.26, 140.03, 138.44, 147.00, 142.44, 139.47, 136.25],
      },
    },
    {
      key: "ex2", date: "2026-04-26", zone: "ZONE_3",
      original: {
        width90: 110.20, mae: 10.02, coverage90: 96.9,
        lo90: [34.02, 29.73, 22.77, 6.59, 22.49, 16.48, 15.31, 4.11, 13.71, 10.45, 8.23, -0.20, 4.09, 3.90, 2.41, -1.99, 1.35, 4.04, 2.87, -2.38, 3.61, 6.12, 5.08, -0.75, 3.07, 4.31, 1.46, -5.92, -0.14, -0.02, -0.56, -7.23, 3.83, 0.22, -6.66, -21.35, -20.63, -27.31, -52.53, -66.38, -75.01, -103.89, -163.27, -181.81, -206.88, -260.70, -356.10, -362.33, -350.11, -414.98, -534.29, -541.08, -512.08, -573.89, -671.91, -632.22, -560.67, -570.39, -604.54, -506.27, -436.05, -387.16, -357.13, -247.19, -183.00, -129.99, -98.10, -47.74, -49.46, -45.36, -37.92, -28.37, -15.82, -15.54, -10.96, -5.56, 2.79, 11.64, 19.04, 22.07, 12.32, 19.82, 30.50, 35.77, 51.39, 58.23, 68.72, 59.71, 78.88, 78.86, 82.75, 68.17, 74.20, 65.82, 65.34, 49.83],
        q25: [47.75, 40.86, 35.69, 23.96, 35.56, 26.90, 26.21, 18.06, 26.54, 20.43, 19.20, 13.29, 16.74, 13.16, 12.76, 10.36, 14.49, 13.50, 13.58, 10.79, 17.20, 16.17, 16.20, 12.75, 17.12, 14.69, 13.11, 8.47, 13.24, 9.88, 10.34, 6.24, 17.37, 11.27, 6.64, -4.14, 1.47, -3.08, -18.46, -32.08, -24.33, -37.05, -74.54, -105.14, -119.70, -148.23, -208.30, -238.40, -245.35, -283.71, -358.25, -393.16, -419.43, -455.80, -506.50, -493.28, -485.94, -477.27, -469.79, -398.03, -366.74, -306.99, -250.73, -162.86, -139.94, -76.67, -34.11, 5.13, -26.68, -15.54, -8.73, 1.70, -0.89, -0.45, 4.43, 11.71, 17.82, 23.46, 31.20, 36.22, 27.63, 32.18, 43.18, 50.93, 67.31, 72.07, 82.06, 77.36, 93.53, 91.25, 94.43, 83.38, 88.28, 77.85, 77.16, 66.10],
        q50: [52.00, 44.34, 39.58, 30.35, 39.28, 30.07, 29.34, 22.88, 30.10, 23.45, 22.29, 17.87, 20.12, 15.98, 15.61, 14.41, 18.04, 16.39, 16.54, 15.17, 20.95, 19.23, 19.33, 17.26, 21.19, 17.89, 16.41, 13.31, 17.23, 12.90, 13.37, 10.62, 22.82, 14.81, 10.98, 1.87, 17.23, 6.23, -4.27, -18.79, 12.34, -11.05, -38.63, -75.40, -61.54, -104.34, -149.43, -192.00, -177.84, -231.40, -289.08, -339.05, -357.99, -407.35, -441.18, -443.67, -437.28, -437.95, -415.11, -358.67, -328.45, -274.25, -206.55, -129.05, -115.14, -54.95, -5.82, 27.59, -14.19, -4.65, 4.44, 14.39, 3.83, 4.19, 9.09, 17.69, 21.91, 27.18, 34.84, 41.30, 32.41, 36.32, 47.12, 56.34, 72.63, 76.45, 86.25, 83.89, 98.09, 95.08, 98.01, 89.13, 92.77, 81.53, 80.83, 72.07],
        q75: [58.83, 48.81, 43.72, 35.20, 44.59, 33.67, 33.02, 27.05, 35.15, 26.91, 25.72, 21.94, 24.60, 19.22, 18.84, 18.23, 22.69, 19.80, 19.93, 19.18, 25.75, 22.73, 22.88, 21.47, 26.53, 21.66, 20.00, 17.47, 22.37, 16.31, 16.72, 14.30, 29.46, 19.33, 15.15, 6.42, 30.97, 18.13, 8.48, -5.34, 42.65, 17.50, -6.86, -42.05, -14.55, -57.91, -97.82, -137.43, -122.96, -175.88, -229.24, -274.97, -307.55, -357.14, -386.38, -384.02, -394.93, -401.22, -373.43, -309.14, -292.06, -248.62, -173.31, -86.66, -91.19, -39.76, 16.54, 55.80, -1.40, 2.73, 16.12, 28.19, 9.02, 7.91, 13.97, 24.70, 26.72, 31.07, 39.26, 46.99, 38.51, 41.28, 52.28, 61.88, 80.31, 81.78, 91.59, 89.45, 104.90, 99.62, 102.77, 94.13, 99.73, 86.17, 85.75, 76.87],
        hi90: [77.96, 62.01, 57.87, 50.80, 60.49, 44.63, 46.41, 41.20, 50.68, 37.59, 38.36, 35.95, 38.99, 29.26, 30.98, 31.66, 37.59, 30.39, 32.58, 33.21, 41.09, 33.60, 36.04, 35.95, 43.08, 33.25, 33.17, 31.72, 38.35, 27.00, 29.17, 27.06, 48.04, 32.46, 27.83, 19.18, 65.09, 49.03, 31.34, 15.56, 119.14, 93.35, 50.03, 9.36, 108.21, 67.54, -2.92, -48.00, 23.04, -24.47, -116.99, -171.00, -171.50, -218.13, -285.24, -291.24, -280.76, -300.12, -299.36, -230.27, -197.77, -180.66, -110.01, -5.58, -31.48, -1.54, 59.14, 110.31, 30.14, 22.99, 42.81, 55.84, 26.09, 20.36, 31.20, 43.93, 42.49, 43.02, 55.05, 65.37, 57.10, 55.43, 69.07, 79.32, 101.93, 97.11, 108.93, 106.72, 124.03, 112.77, 118.74, 110.28, 118.95, 99.71, 102.08, 92.20],
      },
      revin: {
        width90: 115.35, mae: 10.66, coverage90: 99.0,
        lo90: [-7.17, -1.06, -12.76, -54.93, -23.04, -17.75, -18.26, -54.86, -31.65, -23.96, -25.87, -59.98, -42.84, -31.33, -31.21, -59.50, -45.81, -30.56, -30.97, -60.88, -45.21, -28.90, -29.30, -59.87, -44.69, -30.06, -33.31, -67.24, -47.94, -36.04, -34.58, -67.89, -40.07, -34.50, -37.52, -80.81, -44.12, -41.22, -51.56, -101.30, -46.89, -59.77, -84.43, -158.37, -115.62, -149.87, -193.10, -271.36, -235.36, -276.11, -323.68, -400.52, -416.31, -445.84, -470.21, -498.41, -496.75, -471.46, -444.66, -416.79, -393.69, -312.81, -243.59, -197.08, -189.15, -99.56, -48.77, -43.74, -85.88, -51.67, -41.82, -58.12, -71.82, -46.11, -37.16, -52.45, -55.06, -26.07, -13.47, -30.75, -36.91, -13.67, -2.30, -20.32, 4.00, 25.98, 37.30, 4.17, 26.12, 41.25, 53.34, 13.81, 25.93, 30.57, 35.84, -6.77],
        q25: [43.46, 35.14, 23.05, -3.41, 27.09, 17.91, 16.59, -5.41, 19.01, 11.63, 8.96, -11.01, 7.31, 3.60, 2.98, -11.41, 5.02, 4.75, 3.68, -11.90, 6.80, 7.44, 6.21, -10.00, 7.95, 6.76, 2.83, -16.24, 3.66, 0.58, 1.07, -17.48, 10.66, 2.60, -1.30, -28.88, 5.45, -5.19, -16.12, -49.77, 0.95, -24.16, -49.36, -106.21, -70.58, -116.64, -159.57, -221.87, -194.57, -246.00, -293.28, -357.00, -376.39, -418.51, -442.87, -459.22, -455.00, -444.45, -417.59, -378.14, -346.09, -280.98, -213.41, -153.50, -139.01, -63.66, -15.46, 4.60, -34.66, -15.42, -6.87, -8.76, -18.57, -9.10, -1.82, -3.10, 0.37, 12.23, 22.63, 19.62, 16.42, 23.46, 33.26, 29.95, 56.02, 63.49, 73.23, 56.07, 78.09, 78.92, 88.76, 64.95, 76.98, 67.54, 71.03, 43.55],
        q50: [61.13, 48.10, 36.78, 20.26, 43.94, 30.59, 29.80, 17.24, 35.85, 24.20, 22.12, 11.53, 23.70, 15.89, 15.85, 10.69, 21.58, 17.13, 16.73, 10.69, 23.89, 20.20, 19.63, 13.05, 25.42, 19.64, 16.45, 7.28, 20.86, 13.32, 14.40, 5.55, 28.19, 15.46, 12.22, -5.40, 22.92, 7.28, -2.94, -26.84, 18.69, -11.82, -36.32, -83.28, -54.23, -105.24, -147.35, -200.48, -180.38, -235.85, -282.57, -338.58, -364.36, -409.78, -433.64, -442.42, -443.31, -435.98, -408.47, -361.24, -331.77, -270.38, -202.28, -133.87, -121.80, -50.84, -2.23, 26.54, -17.60, -2.75, 6.49, 13.84, -1.15, 4.01, 11.85, 20.09, 18.64, 25.97, 36.80, 43.81, 34.38, 36.87, 47.19, 53.74, 74.36, 77.11, 87.34, 80.40, 96.30, 92.54, 102.61, 88.86, 94.80, 80.86, 84.67, 66.87],
        q75: [85.77, 63.30, 50.47, 37.03, 66.89, 44.68, 43.05, 33.39, 58.77, 38.14, 35.02, 27.33, 45.69, 29.35, 28.29, 26.15, 43.70, 30.69, 29.15, 26.17, 46.27, 33.88, 32.23, 28.78, 48.22, 33.48, 29.04, 23.01, 43.04, 26.68, 26.77, 20.97, 50.95, 29.06, 24.75, 10.22, 45.30, 20.60, 9.18, -11.56, 41.57, 1.34, -24.44, -67.99, -32.68, -93.00, -136.47, -186.42, -162.04, -225.29, -273.23, -326.45, -348.33, -400.61, -425.73, -431.95, -427.84, -426.94, -400.52, -350.35, -314.33, -259.56, -191.81, -119.12, -102.51, -37.99, 10.65, 44.93, 3.20, 10.63, 19.09, 30.78, 20.66, 17.85, 24.93, 36.98, 42.03, 40.49, 50.55, 61.23, 57.96, 51.61, 60.99, 70.78, 98.77, 92.31, 101.72, 97.90, 120.43, 107.58, 117.24, 106.59, 119.06, 95.99, 99.15, 83.81],
        hi90: [139.78, 102.76, 90.03, 88.04, 117.64, 81.74, 81.93, 84.04, 109.40, 74.84, 73.14, 77.75, 94.50, 65.00, 65.49, 76.32, 92.60, 66.54, 66.38, 76.66, 95.51, 69.91, 69.79, 79.74, 98.16, 69.80, 66.53, 73.66, 91.75, 61.87, 63.71, 70.41, 101.06, 64.93, 61.78, 58.23, 94.84, 55.75, 44.91, 34.15, 92.21, 36.18, 10.05, -24.79, 15.45, -60.27, -104.56, -146.28, -120.53, -196.76, -245.33, -290.84, -312.20, -375.70, -400.58, -396.54, -393.46, -402.50, -374.70, -311.58, -276.09, -231.24, -159.43, -70.60, -59.76, -5.21, 47.94, 98.96, 49.62, 45.60, 56.25, 82.76, 68.54, 53.73, 63.19, 90.08, 92.12, 77.50, 90.34, 116.51, 108.57, 89.03, 100.66, 124.30, 151.28, 130.92, 142.42, 150.71, 172.63, 146.00, 158.82, 160.14, 172.04, 134.90, 140.36, 135.27],
      },
    },
  ];

  // illustrative_quantile_bands_20261017/example_predictions.csv - one day of
  // the original champion where calm and volatile stretches share a day, zone
  // and price scale. Window widths from that study's summary.json; `from`/`to`
  // are quarter indices (12:00-15:00 and 21:00-24:00).
  const BAND_CONTRAST = {
    date: "2026-06-11", zone: "ZONE_3",
    windows: [
      { from: 48, to: 60, label: "Calm · width 14.5" },
      { from: 84, to: 96, label: "Volatile · width 27.0" },
    ],
    bands: {
        lo90: [35.98, 26.56, 16.42, -6.17, 6.99, -2.51, 1.48, -4.32, 12.81, 12.91, 16.37, 5.61, -0.97, -6.12, -7.06, -12.00, -9.16, -7.57, -2.91, -4.13, 5.61, 9.08, 8.91, 2.61, -2.09, -1.61, 2.97, 5.47, 31.13, 39.20, 42.05, 28.96, 29.11, 15.62, 6.85, -10.18, -1.93, -8.30, -9.11, -15.23, -10.17, -8.71, -6.07, -9.24, -9.85, -7.35, -5.75, -8.64, -8.38, -6.65, -6.00, -9.39, -7.90, -6.04, -6.13, -9.47, -8.73, -6.19, -6.16, -8.66, -9.51, -6.90, -6.61, -8.71, -10.46, -7.88, -8.14, -10.83, -12.46, -10.12, -9.41, -11.02, -11.55, -7.35, -3.99, -2.93, 2.69, 9.18, 12.57, 10.67, 4.62, 4.94, 5.53, -3.54, 2.80, -2.49, 0.53, -11.23, 1.64, 1.32, 13.75, 8.16, 9.38, 5.69, 13.15, 2.78],
        q25: [46.45, 36.10, 26.06, 7.35, 15.53, 5.95, 8.47, 4.96, 20.66, 20.35, 22.77, 13.74, 7.06, 0.77, -0.63, -3.71, -0.30, -0.02, 3.87, 4.76, 14.10, 16.18, 15.85, 11.86, 8.04, 7.11, 11.11, 16.27, 41.20, 47.98, 50.04, 39.30, 39.22, 24.57, 15.35, 0.69, 6.83, -1.20, -2.81, -7.22, -2.65, -2.88, -0.86, -2.50, -2.90, -2.11, -0.88, -2.29, -1.69, -1.72, -1.30, -3.11, -1.38, -1.29, -1.51, -3.34, -2.24, -1.42, -1.50, -2.41, -2.91, -1.84, -1.56, -1.87, -2.98, -1.86, -2.10, -2.80, -3.84, -3.03, -2.54, -1.91, -2.28, 0.48, 3.57, 6.97, 13.01, 18.20, 21.06, 21.39, 16.48, 15.53, 15.31, 9.57, 14.77, 8.59, 9.98, 1.58, 12.07, 11.82, 21.53, 18.01, 18.84, 14.96, 20.27, 12.40],
        q50: [51.25, 39.53, 29.69, 13.15, 18.91, 8.85, 10.95, 8.80, 23.69, 22.96, 25.04, 17.12, 9.94, 3.13, 1.63, -0.25, 2.78, 2.52, 6.23, 8.48, 17.14, 18.60, 18.29, 15.70, 11.66, 10.02, 13.99, 20.81, 45.01, 50.99, 52.91, 43.72, 43.21, 27.67, 18.42, 5.34, 9.85, 1.17, -0.66, -3.91, -0.20, -0.97, 0.90, 0.25, -0.70, -0.41, 0.76, 0.32, 0.37, -0.14, 0.26, -0.54, 0.58, 0.22, 0.00, -0.84, -0.29, 0.10, 0.03, 0.12, -0.87, -0.20, 0.13, 0.90, -0.60, 0.11, -0.04, 0.49, -1.01, -0.70, -0.17, 1.89, 0.86, 3.07, 6.21, 11.13, 16.58, 21.23, 24.07, 25.99, 21.07, 19.12, 18.83, 15.11, 19.53, 12.36, 13.34, 6.99, 16.15, 15.42, 24.27, 22.15, 22.60, 18.15, 22.79, 16.45],
        q75: [57.65, 43.39, 33.30, 17.19, 22.63, 11.34, 13.88, 11.86, 26.92, 25.19, 27.59, 19.93, 13.17, 5.34, 3.87, 2.44, 6.10, 4.83, 8.64, 11.37, 20.60, 20.98, 20.54, 18.52, 15.56, 12.80, 16.78, 24.12, 49.19, 53.74, 55.72, 46.91, 48.28, 30.45, 21.12, 8.42, 13.22, 3.16, 1.31, -1.65, 2.40, 0.69, 2.57, 2.31, 1.70, 1.15, 2.28, 2.27, 2.78, 1.39, 1.71, 1.32, 2.89, 1.69, 1.36, 0.93, 1.94, 1.56, 1.39, 1.96, 1.38, 1.32, 1.64, 3.00, 2.00, 1.94, 1.78, 2.95, 2.05, 1.47, 1.98, 4.71, 4.21, 5.54, 8.70, 14.34, 20.35, 23.93, 26.84, 29.64, 25.74, 22.57, 22.38, 19.20, 24.74, 15.72, 16.99, 10.66, 19.99, 18.01, 27.80, 25.55, 26.51, 20.65, 26.07, 19.55],
        hi90: [72.37, 54.39, 43.27, 26.95, 32.05, 18.90, 22.99, 19.89, 35.17, 31.78, 35.40, 27.48, 21.48, 11.84, 10.84, 10.20, 14.65, 11.70, 16.34, 19.95, 29.57, 27.93, 27.58, 26.79, 25.57, 21.09, 25.64, 33.83, 59.65, 61.73, 64.34, 56.01, 60.29, 38.79, 29.33, 16.84, 21.89, 9.24, 7.78, 5.28, 9.19, 5.66, 8.08, 8.70, 7.95, 5.75, 7.25, 8.46, 9.01, 5.90, 6.47, 7.35, 8.92, 6.07, 5.82, 6.77, 7.81, 5.86, 5.90, 7.99, 7.27, 5.76, 6.58, 9.72, 8.82, 7.31, 7.62, 10.71, 10.04, 7.89, 8.90, 13.44, 12.93, 12.82, 16.66, 23.94, 30.09, 31.82, 35.67, 40.51, 37.61, 32.78, 33.33, 30.42, 37.57, 25.95, 28.29, 20.53, 29.84, 25.97, 38.68, 34.62, 36.21, 28.26, 36.11, 27.74],
    },
  };

  let charts = [];

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function zoneLabel(zone) {
    return zone.replace("ZONE_", "Zone ");
  }

  function niceTicks(min, max, targetCount) {
    if (min === max) { min -= 1; max += 1; }
    const span = max - min;
    const rawStep = span / targetCount;
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const residual = rawStep / mag;
    let step;
    if (residual > 5) step = 10 * mag;
    else if (residual > 2) step = 5 * mag;
    else if (residual > 1) step = 2 * mag;
    else step = mag;
    const niceMin = Math.floor(min / step) * step;
    const niceMax = Math.ceil(max / step) * step;
    const ticks = [];
    for (let v = niceMin; v <= niceMax + step * 0.001; v += step) {
      ticks.push(Math.round(v / step) * step);
    }
    return { ticks, niceMin, niceMax, step };
  }

  // Shared canvas plumbing for one chart: DPR-aware sizing from its wrap,
  // tooltip placement, and pointer coordinates. Same as experiment.js's.
  function makeSurface(ids) {
    const wrapEl = document.getElementById(ids.wrap);
    const canvas = document.getElementById(ids.canvas);
    const tooltipEl = document.getElementById(ids.tooltip);
    const ctx = canvas.getContext("2d");
    const surface = { wrapEl, canvas, tooltipEl, ctx, width: 0, height: 0 };

    surface.resize = function () {
      const rect = wrapEl.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      surface.width = rect.width;
      surface.height = rect.height;
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);
    };

    surface.showTooltip = function (pxX, pxY, title, rows) {
      tooltipEl.innerHTML = "";
      const head = document.createElement("div");
      head.className = "tooltip-time";
      head.textContent = title;
      tooltipEl.appendChild(head);
      for (const r of rows) {
        const row = document.createElement("div");
        row.className = "tooltip-row";
        const key = document.createElement("span");
        key.className = "key";
        key.style.background = r.color;
        const name = document.createElement("span");
        name.className = "series-name";
        name.textContent = r.name + ":";
        const val = document.createElement("span");
        val.className = "value";
        val.textContent = r.value;
        row.appendChild(key);
        row.appendChild(name);
        row.appendChild(val);
        tooltipEl.appendChild(row);
      }
      tooltipEl.hidden = false;
      const wrapRect = wrapEl.getBoundingClientRect();
      const ttRect = tooltipEl.getBoundingClientRect();
      let left = pxX + 14;
      let top = pxY + 14;
      if (left + ttRect.width > wrapRect.width) left = pxX - ttRect.width - 14;
      if (top + ttRect.height > wrapRect.height) top = pxY - ttRect.height - 14;
      tooltipEl.style.left = Math.max(0, left) + "px";
      tooltipEl.style.top = Math.max(0, top) + "px";
    };

    surface.hideTooltip = function () {
      tooltipEl.hidden = true;
    };

    surface.pointer = function (evt) {
      const rect = canvas.getBoundingClientRect();
      return { x: evt.clientX - rect.left, y: evt.clientY - rect.top };
    };

    return surface;
  }

  // One line per series over an ordered categorical x axis. Each series is
  // {name, values, colorVar, dotted}: solid 2px for a primary series, dotted
  // 1.5px faded for a contextual one (the same-hue secondary style). Lines end
  // in direct labels, de-collided with a 13px minimum gap.
  function seriesLineChart(spec) {
    const s = makeSurface(spec.ids);
    const MARGIN = { top: 14, right: spec.labelRoom || 64, bottom: 26, left: 44 };
    const n = spec.labels.length;
    let layout = null;

    function render(hoverIdx) {
      s.resize();
      const { ctx } = s;
      const plot = {
        left: MARGIN.left,
        top: MARGIN.top,
        width: s.width - MARGIN.left - MARGIN.right,
        height: s.height - MARGIN.top - MARGIN.bottom,
      };
      if (plot.width <= 0 || plot.height <= 0) { layout = null; return; }

      let lo = Infinity, hi = -Infinity;
      for (const ser of spec.series) for (const v of ser.values) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      const { ticks, niceMin, niceMax } = niceTicks(lo, hi, 4);
      const band = plot.width / n;
      const idxToX = (i) => plot.left + band * (i + 0.5);
      const valueToY = (v) => plot.top + (1 - (v - niceMin) / (niceMax - niceMin)) * plot.height;

      ctx.font = FONT;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (const t of ticks) {
        const y = Math.round(valueToY(t)) + 0.5;
        ctx.strokeStyle = cssVar("--grid-line");
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(plot.left, y);
        ctx.lineTo(plot.left + plot.width, y);
        ctx.stroke();
        ctx.fillStyle = cssVar("--text-muted");
        ctx.fillText(t.toFixed(spec.tickDecimals), plot.left - 10, y);
      }

      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let i = 0; i < n; i++) {
        ctx.fillStyle = cssVar("--text-muted");
        ctx.fillText(spec.labels[i], idxToX(i), plot.top + plot.height + 8);
      }
      ctx.strokeStyle = cssVar("--axis-line");
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(plot.left, plot.top + plot.height + 0.5);
      ctx.lineTo(plot.left + plot.width, plot.top + plot.height + 0.5);
      ctx.stroke();

      if (hoverIdx !== null && hoverIdx !== undefined) {
        const x = Math.round(idxToX(hoverIdx)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(x, plot.top);
        ctx.lineTo(x, plot.top + plot.height);
        ctx.stroke();
      }

      const surfaceColor = cssVar("--surface-1");
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      for (const ser of spec.series) {
        const color = cssVar(ser.colorVar);
        const alpha = ser.dotted ? 0.55 : 1;
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = ser.dotted ? 1.5 : 2;
        if (ser.dotted) ctx.setLineDash([1, 4]);
        ctx.globalAlpha = alpha;
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
          if (i === 0) ctx.moveTo(idxToX(i), valueToY(ser.values[i]));
          else ctx.lineTo(idxToX(i), valueToY(ser.values[i]));
        }
        ctx.stroke();
        ctx.restore();

        // Markers, ringed in the surface colour so overlapping series stay
        // separable.
        for (let i = 0; i < n; i++) {
          const x = idxToX(i), y = valueToY(ser.values[i]);
          const r = i === hoverIdx ? 5 : 4;
          ctx.globalAlpha = alpha;
          ctx.beginPath();
          ctx.fillStyle = surfaceColor;
          ctx.arc(x, y, r + 2, 0, Math.PI * 2);
          ctx.fill();
          ctx.beginPath();
          ctx.fillStyle = color;
          ctx.arc(x, y, r, 0, Math.PI * 2);
          ctx.fill();
          ctx.globalAlpha = 1;
        }
      }

      const ends = spec.series
        .map((ser) => ({ name: ser.name, y: valueToY(ser.values[n - 1]) }))
        .sort((a, b) => a.y - b.y);
      for (let i = 1; i < ends.length; i++) {
        if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13;
      }
      ctx.font = FONT;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillStyle = cssVar("--text-secondary");
      for (const e of ends) ctx.fillText(e.name, idxToX(n - 1) + 14, e.y);

      layout = { plot, band };
    }

    function handleMove(evt) {
      if (!layout) return;
      const p = s.pointer(evt);
      const { plot, band } = layout;
      if (p.x < plot.left || p.x > plot.left + plot.width || p.y < plot.top || p.y > plot.top + plot.height) {
        s.hideTooltip();
        render(null);
        return;
      }
      const idx = Math.min(n - 1, Math.floor((p.x - plot.left) / band));
      render(idx);
      s.showTooltip(p.x, p.y, spec.tipTitle(idx), spec.series.map((ser) => ({
        name: ser.name,
        value: ser.values[idx].toFixed(spec.valueDecimals),
        color: cssVar(ser.colorVar),
      })));
    }

    function handleLeave() {
      s.hideTooltip();
      render(null);
    }

    s.canvas.addEventListener("mousemove", handleMove);
    s.canvas.addEventListener("mouseleave", handleLeave);
    return { render: () => render(null), wrapEl: s.wrapEl };
  }

  function quarterTime(i) {
    const fmt = (m) => String(Math.floor(m / 60)).padStart(2, "0") + ":" + String(m % 60).padStart(2, "0");
    return fmt(i * 15) + "–" + fmt(i * 15 + 15);
  }

  // One day of quantile bands for one zone, over its 96 quarters, all in the
  // zone's hue: calibrated 90% interval as a light fill, raw q25-q75 as a
  // darker one, q50 as the solid primary line and the observed price as the
  // dotted secondary line. Observed quarters outside the 90% interval get a
  // marker. `yDomain` is passed in so paired panels share one scale.
  // Optional `windows` ({from, to, label}, quarter indices) shade and label
  // stretches the prose refers to, in neutral ink behind the data.
  function bandChart(spec) {
    const s = makeSurface(spec.ids);
    const MARGIN = { top: 12, right: 22, bottom: 26, left: 44 };
    const b = spec.bands;
    const obs = spec.observed;
    const n = obs.length;
    let layout = null;

    function fillBand(ctx, lo, hi, idxToX, valueToY) {
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        if (i === 0) ctx.moveTo(idxToX(i), valueToY(hi[i]));
        else ctx.lineTo(idxToX(i), valueToY(hi[i]));
      }
      for (let i = n - 1; i >= 0; i--) ctx.lineTo(idxToX(i), valueToY(lo[i]));
      ctx.closePath();
      ctx.fill();
    }

    function render(hoverIdx) {
      s.resize();
      const { ctx } = s;
      const plot = {
        left: MARGIN.left,
        top: MARGIN.top,
        width: s.width - MARGIN.left - MARGIN.right,
        height: s.height - MARGIN.top - MARGIN.bottom,
      };
      if (plot.width <= 0 || plot.height <= 0) { layout = null; return; }

      const { ticks, niceMin, niceMax } = niceTicks(spec.yDomain.lo, spec.yDomain.hi, 4);
      const idxToX = (i) => plot.left + ((i + 0.5) / n) * plot.width;
      const valueToY = (v) => plot.top + (1 - (v - niceMin) / (niceMax - niceMin)) * plot.height;

      ctx.font = FONT;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (const t of ticks) {
        const y = Math.round(valueToY(t)) + 0.5;
        ctx.strokeStyle = t === 0 ? cssVar("--axis-line") : cssVar("--grid-line");
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(plot.left, y);
        ctx.lineTo(plot.left + plot.width, y);
        ctx.stroke();
        ctx.fillStyle = cssVar("--text-muted");
        ctx.fillText(String(t), plot.left - 10, y);
      }

      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = cssVar("--text-muted");
      for (let h = 0; h <= 24; h += 6) {
        ctx.fillText(String(h).padStart(2, "0") + ":00", plot.left + (h / 24) * plot.width, plot.top + plot.height + 8);
      }
      ctx.strokeStyle = cssVar("--axis-line");
      ctx.beginPath();
      ctx.moveTo(plot.left, plot.top + plot.height + 0.5);
      ctx.lineTo(plot.left + plot.width, plot.top + plot.height + 0.5);
      ctx.stroke();

      // Labels sit centred over their window, kept inside the plot; when any
      // two would touch, all of them drop their detail after " · ".
      const windows = (spec.windows || []).map((w) => ({
        ...w,
        x0: plot.left + (w.from / n) * plot.width,
        x1: plot.left + (w.to / n) * plot.width,
      }));
      const placeAll = (short) => windows.map((w) => {
        const text = short ? w.label.split(" · ")[0] : w.label;
        const tw = ctx.measureText(text).width;
        const x = Math.max(plot.left, Math.min((w.x0 + w.x1) / 2 - tw / 2, plot.left + plot.width - tw));
        return { text, x, right: x + tw };
      });
      let labels = placeAll(false);
      if (labels.some((l, i) => i > 0 && l.x < labels[i - 1].right + 8)) labels = placeAll(true);
      windows.forEach((w, i) => {
        ctx.fillStyle = cssVar("--text-muted");
        ctx.globalAlpha = 0.08;
        ctx.fillRect(w.x0, plot.top, w.x1 - w.x0, plot.height);
        ctx.globalAlpha = 1;
        ctx.fillStyle = cssVar("--text-secondary");
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillText(labels[i].text, labels[i].x, plot.top + 4);
      });

      const color = cssVar(ZONE_COLOR_VAR[spec.zone]);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.16;
      fillBand(ctx, b.lo90, b.hi90, idxToX, valueToY);
      ctx.globalAlpha = 0.3;
      fillBand(ctx, b.q25, b.q75, idxToX, valueToY);
      ctx.globalAlpha = 1;

      if (hoverIdx !== null && hoverIdx !== undefined) {
        const x = Math.round(idxToX(hoverIdx)) + 0.5;
        ctx.strokeStyle = cssVar("--axis-line");
        ctx.beginPath();
        ctx.moveTo(x, plot.top);
        ctx.lineTo(x, plot.top + plot.height);
        ctx.stroke();
      }

      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.strokeStyle = color;

      // Observed: dotted 1.5px, faded - the same-hue secondary style.
      ctx.save();
      ctx.lineWidth = 1.5;
      ctx.setLineDash([1, 4]);
      ctx.globalAlpha = 0.55;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        if (i === 0) ctx.moveTo(idxToX(i), valueToY(obs[i]));
        else ctx.lineTo(idxToX(i), valueToY(obs[i]));
      }
      ctx.stroke();
      ctx.restore();

      // q50: solid 2px.
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        if (i === 0) ctx.moveTo(idxToX(i), valueToY(b.q50[i]));
        else ctx.lineTo(idxToX(i), valueToY(b.q50[i]));
      }
      ctx.stroke();

      // Misses: observed outside the calibrated 90% interval, ringed in the
      // surface colour.
      const surfaceColor = cssVar("--surface-1");
      for (let i = 0; i < n; i++) {
        if (obs[i] >= b.lo90[i] && obs[i] <= b.hi90[i] && i !== hoverIdx) continue;
        const x = idxToX(i), y = valueToY(obs[i]);
        const r = i === hoverIdx ? 4 : 3;
        ctx.beginPath();
        ctx.fillStyle = surfaceColor;
        ctx.arc(x, y, r + 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.fillStyle = color;
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
      }

      layout = { plot };
    }

    function handleMove(evt) {
      if (!layout) return;
      const p = s.pointer(evt);
      const { plot } = layout;
      if (p.x < plot.left || p.x > plot.left + plot.width || p.y < plot.top || p.y > plot.top + plot.height) {
        s.hideTooltip();
        render(null);
        return;
      }
      const idx = Math.min(n - 1, Math.floor(((p.x - plot.left) / plot.width) * n));
      render(idx);
      const color = cssVar(ZONE_COLOR_VAR[spec.zone]);
      const inside = obs[idx] >= b.lo90[idx] && obs[idx] <= b.hi90[idx];
      s.showTooltip(p.x, p.y, quarterTime(idx), [
        { name: "Observed", value: obs[idx].toFixed(2) + (inside ? "" : " (outside 90%)"), color },
        { name: "q50", value: b.q50[idx].toFixed(2), color },
        { name: "90% interval", value: b.lo90[idx].toFixed(2) + " to " + b.hi90[idx].toFixed(2), color },
      ]);
    }

    function handleLeave() {
      s.hideTooltip();
      render(null);
    }

    s.canvas.addEventListener("mousemove", handleMove);
    s.canvas.addEventListener("mouseleave", handleLeave);
    return { render: () => render(null), wrapEl: s.wrapEl };
  }

  function observedDay(date, zone) {
    const dayIdx = DATASET.dates.indexOf(date);
    return Array.from(DATASET.prices[zone].subarray(dayIdx * 96, dayIdx * 96 + 96));
  }

  function buildLegend(el) {
    el.innerHTML = "";
    for (const zone of DATASET.zones) {
      const item = document.createElement("span");
      item.className = "legend-item";
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = `var(${ZONE_COLOR_VAR[zone]})`;
      item.appendChild(swatch);
      item.appendChild(document.createTextNode(zoneLabel(zone)));
      el.appendChild(item);
    }
  }

  function monthTitle(i) {
    return MONTHS.labels[i] + " 2026 (" + MONTHS.trainMonths[i] + " training months)";
  }

  function render() {
    for (const c of charts) c.render();
  }

  function init(opts) {
    opts.root.querySelectorAll(".ens-zone-legend").forEach(buildLegend);

    charts = [
      seriesLineChart({
        ids: { wrap: "ens-month-wrap", canvas: "ens-month-canvas", tooltip: "ens-month-tooltip" },
        labels: MONTHS.labels,
        series: [
          { name: "Ensemble", values: MONTHLY_MAE.ensemble, colorVar: "--series-pooled", dotted: false },
          { name: "Interpolation", values: MONTHLY_MAE.interpolation, colorVar: "--series-pooled", dotted: true },
        ],
        labelRoom: 92,
        tickDecimals: 0,
        valueDecimals: 2,
        tipTitle: monthTitle,
      }),
      seriesLineChart({
        ids: { wrap: "ens-zone-wrap", canvas: "ens-zone-canvas", tooltip: "ens-zone-tooltip" },
        labels: MONTHS.labels,
        series: DATASET.zones.map((z) => ({
          name: zoneLabel(z), values: ZONE_RATIO[z], colorVar: ZONE_COLOR_VAR[z], dotted: false,
        })),
        tickDecimals: 1,
        valueDecimals: 3,
        tipTitle: monthTitle,
      }),
      seriesLineChart({
        ids: { wrap: "ens-revin-wrap", canvas: "ens-revin-canvas", tooltip: "ens-revin-tooltip" },
        labels: MONTHS.labels,
        series: [
          { name: "Original", values: REVIN_RATIO.original, colorVar: "--series-pooled", dotted: false },
          { name: "RevIN", values: REVIN_RATIO.revin, colorVar: "--series-pooled", dotted: true },
        ],
        tickDecimals: 2,
        valueDecimals: 4,
        tipTitle: monthTitle,
      }),
    ];

    // Paired panels (original vs RevIN) of one example day share a y scale.
    for (const ex of EXAMPLES) {
      const observed = observedDay(ex.date, ex.zone);
      let lo = Math.min(...observed), hi = Math.max(...observed);
      for (const arm of [ex.original, ex.revin]) {
        lo = Math.min(lo, ...arm.lo90);
        hi = Math.max(hi, ...arm.hi90);
      }
      for (const armKey of ["original", "revin"]) {
        const id = "ens-" + ex.key + "-" + armKey;
        charts.push(bandChart({
          ids: { wrap: id + "-wrap", canvas: id + "-canvas", tooltip: id + "-tooltip" },
          zone: ex.zone,
          bands: ex[armKey],
          observed,
          yDomain: { lo, hi },
        }));
      }
    }

    const contrastObserved = observedDay(BAND_CONTRAST.date, BAND_CONTRAST.zone);
    const contrastBands = BAND_CONTRAST.bands;
    charts.push(bandChart({
      ids: { wrap: "ens-contrast-wrap", canvas: "ens-contrast-canvas", tooltip: "ens-contrast-tooltip" },
      zone: BAND_CONTRAST.zone,
      bands: contrastBands,
      observed: contrastObserved,
      yDomain: {
        lo: Math.min(...contrastObserved, ...contrastBands.lo90),
        hi: Math.max(...contrastObserved, ...contrastBands.hi90),
      },
      windows: BAND_CONTRAST.windows,
    }));

    let resizeTimer = null;
    const ro = new ResizeObserver(() => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(render, 150);
    });
    for (const c of charts) ro.observe(c.wrapEl);

    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
  }

  return { init, render };
})();
