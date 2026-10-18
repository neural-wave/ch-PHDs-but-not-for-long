AET Neural Wave 2026 — participant training package

This README is bilingual. The English version comes first; the Italian version follows below.

Contents
- train_prices_15min.csv: observed day-ahead electricity prices at 15-minute resolution.
- Columns: delivery_date, market_id, hour_index, quarter_index, price_eur_mwh.
- market_id takes the values ZONE_0, ZONE_1, ZONE_2 and ZONE_3. These labels remain consistent across all official challenge files.
- delivery_date is the delivery date in the CET/CEST calendar; hour_index ranges from 1 to 24 and quarter_index from 1 to 4.
- Prices are expressed in EUR/MWh and provided to two decimal places.

Coverage
- Delivery dates run from 1 October 2025 to 30 September 2026, excluding the two daylight-saving-time change dates.
- The dataset contains 363 valid days and 34,848 quarter-hourly prices per zone, for 139,392 rows in total.
- For each zone and training hour, the reference hourly price is the arithmetic mean of the four quarter-hourly prices.

Test data and input rules
- This package contains training data only. Targets for evaluation dates are not included.
- After the solution freeze, for each evaluation date the organizers will provide test_hourly.csv with the 24 hourly prices per zone and history_15min.csv with quarter-hourly prices from strictly earlier dates.
- Expected format of test_hourly.csv: delivery_date, market_id, hour_index, hourly_price_eur_mwh.
- Expected format of history_15min.csv: delivery_date, market_id, hour_index, quarter_index, price_eur_mwh.
- Recent history supplied after the freeze may only be used by inference rules fixed in advance. It does not authorize retraining, recalibration, hyperparameter selection or changes to the model.
- Before the freeze, participants must not access or use delivery dates from 1 to 18 October 2026, even if they are publicly available.
- External data and models are not permitted; only the official challenge package may be used.

Public price data reference: ENTSO-E Transparency Platform, https://transparency.entsoe.eu/.

============================================================
Versione italiana
============================================================

AET Neural Wave 2026 — pacchetto di training per i partecipanti

Contenuto
- train_prices_15min.csv: prezzi day-ahead osservati ogni 15 minuti.
- Colonne: delivery_date, market_id, hour_index, quarter_index, price_eur_mwh.
- market_id assume solo i valori ZONE_0, ZONE_1, ZONE_2 e ZONE_3. Le etichette sono stabili in tutti i file ufficiali della challenge.
- delivery_date è la data di consegna nel calendario CET/CEST; hour_index va da 1 a 24 e quarter_index da 1 a 4.
- I prezzi sono espressi in EUR/MWh e riportati con due cifre decimali.

Copertura
- Delivery date dal 1° ottobre 2025 al 30 settembre 2026, esclusi i due giorni di cambio dell'ora.
- 363 giornate valide e 34.848 prezzi quartorari per zona; 139.392 righe in totale.
- Per ogni zona e ora del training, il prezzo orario di riferimento è la media aritmetica dei quattro prezzi quartorari.

Test e regole sugli input
- Il pacchetto contiene soltanto il training. I target delle date di valutazione non sono inclusi.
- Dopo il freeze, per ogni data valutata la giuria fornirà test_hourly.csv con i 24 prezzi orari per zona e history_15min.csv con prezzi quartorari di date strettamente precedenti.
- Formato previsto di test_hourly.csv: delivery_date, market_id, hour_index, hourly_price_eur_mwh.
- Formato previsto di history_15min.csv: delivery_date, market_id, hour_index, quarter_index, price_eur_mwh.
- Lo storico recente fornito dopo il freeze può alimentare soltanto regole di inferenza già fissate. Non autorizza riaddestramento, calibrazione, selezione di iperparametri o modifiche del modello.
- Prima del freeze, le delivery date dal 1° al 18 ottobre 2026 non possono essere consultate o utilizzate dai partecipanti, anche se pubblicamente disponibili.
- Non sono ammessi dati o modelli esterni al pacchetto ufficiale.

Riferimento pubblico per i prezzi: ENTSO-E Transparency Platform, https://transparency.entsoe.eu/.
