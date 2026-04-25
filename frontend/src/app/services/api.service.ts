import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import {
  PurchaseClassificationRequest,
  PurchaseClassificationResponse,
  PurchaseRegressionRequest,
  PurchaseRegressionResponse,
  PurchaseClusteringRequest,
  PurchaseClusteringResponse,
  CommercialRegressionRequest,
  CommercialRegressionResponse,
  CommercialClassificationRequest,
  CommercialClassificationResponse,
  CommercialAnomalyRequest,
  AnomalyResponse,
  ForecastPeriodRequest,
  CommercialForecastResponse,
  MarketingClusteringRequest,
  MarketingClusteringResponse,
  MarketingTimeseriesResponse,
  MarketingRegressionRequest,
  MarketingRegressionResponse,
  MarketingClassificationRequest,
  MarketingClassificationResponse,
  GmClassificationRequest,
  GmClassificationResponse,
  GmRegressionRequest,
  GmRegressionResponse,
  GmClusteringRequest,
  GmClusteringResponse,
  GmAnomalyRequest,
  B2bFeaturesRequest,
  B2bClassificationResponse,
  B2bClusteringResponse,
  B2bRegressionResponse,
  B2bAnomalyResponse,
  FinClusteringRequest,
  FinClusteringResponse,
  FinForecastResponse,
} from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private base = environment.apiUrl;

  constructor(private http: HttpClient) {}

  // PURCHASE
  purchaseClassify(body: PurchaseClassificationRequest): Observable<PurchaseClassificationResponse> {
    return this.http
      .post<PurchaseClassificationResponse>(`${this.base}/decideur-purchase/classification/predict`, body)
      .pipe(catchError(this.handleError));
  }

  purchaseRegress(body: PurchaseRegressionRequest): Observable<PurchaseRegressionResponse> {
    return this.http
      .post<PurchaseRegressionResponse>(`${this.base}/decideur-purchase/regression/predict`, body)
      .pipe(catchError(this.handleError));
  }

  purchaseCluster(body: PurchaseClusteringRequest): Observable<PurchaseClusteringResponse> {
    return this.http
      .post<PurchaseClusteringResponse>(`${this.base}/decideur-purchase/clustering/predict`, body)
      .pipe(catchError(this.handleError));
  }

  // COMMERCIAL B2C
  commercialRegress(body: CommercialRegressionRequest): Observable<CommercialRegressionResponse> {
    return this.http
      .post<CommercialRegressionResponse>(`${this.base}/decideur-commercial/regression/predict`, body)
      .pipe(catchError(this.handleError));
  }

  commercialClassify(body: CommercialClassificationRequest): Observable<CommercialClassificationResponse> {
    return this.http
      .post<CommercialClassificationResponse>(`${this.base}/decideur-commercial/classification/predict`, body)
      .pipe(catchError(this.handleError));
  }

  commercialAnomaly(body: CommercialAnomalyRequest): Observable<AnomalyResponse> {
    return this.http
      .post<AnomalyResponse>(`${this.base}/decideur-commercial/anomaly/detect`, body)
      .pipe(catchError(this.handleError));
  }

  commercialForecast(body: ForecastPeriodRequest): Observable<CommercialForecastResponse> {
    return this.http
      .post<CommercialForecastResponse>(`${this.base}/decideur-commercial/forecast/predict`, body)
      .pipe(catchError(this.handleError));
  }

  // MARKETING
  marketingCluster(body: MarketingClusteringRequest): Observable<MarketingClusteringResponse> {
    return this.http
      .post<MarketingClusteringResponse>(`${this.base}/decideur-marketing/clustering/predict`, body)
      .pipe(catchError(this.handleError));
  }

  marketingTimeseries(body: ForecastPeriodRequest): Observable<MarketingTimeseriesResponse> {
    return this.http
      .post<MarketingTimeseriesResponse>(`${this.base}/decideur-marketing/timeseries/forecast`, body)
      .pipe(catchError(this.handleError));
  }

  marketingRegress(body: MarketingRegressionRequest): Observable<MarketingRegressionResponse> {
    return this.http
      .post<MarketingRegressionResponse>(`${this.base}/decideur-marketing/regression/predict`, body)
      .pipe(catchError(this.handleError));
  }

  marketingClassify(body: MarketingClassificationRequest): Observable<MarketingClassificationResponse> {
    return this.http
      .post<MarketingClassificationResponse>(`${this.base}/decideur-marketing/classification/predict`, body)
      .pipe(catchError(this.handleError));
  }

  // GM
  gmClassify(body: GmClassificationRequest): Observable<GmClassificationResponse> {
    return this.http
      .post<GmClassificationResponse>(`${this.base}/decideur-gm/classification/predict`, body)
      .pipe(catchError(this.handleError));
  }

  gmRegress(body: GmRegressionRequest): Observable<GmRegressionResponse> {
    return this.http
      .post<GmRegressionResponse>(`${this.base}/decideur-gm/regression/predict`, body)
      .pipe(catchError(this.handleError));
  }

  gmCluster(body: GmClusteringRequest): Observable<GmClusteringResponse> {
    return this.http
      .post<GmClusteringResponse>(`${this.base}/decideur-gm/clustering/predict`, body)
      .pipe(catchError(this.handleError));
  }

  gmAnomaly(body: GmAnomalyRequest): Observable<AnomalyResponse> {
    return this.http
      .post<AnomalyResponse>(`${this.base}/decideur-gm/anomaly/detect`, body)
      .pipe(catchError(this.handleError));
  }

  // B2B
  b2bAnomaly(body: B2bFeaturesRequest): Observable<B2bAnomalyResponse> {
    return this.http
      .post<B2bAnomalyResponse>(`${this.base}/decideur-b2b/anomaly/detect`, body)
      .pipe(catchError(this.handleError));
  }

  b2bClassify(body: B2bFeaturesRequest): Observable<B2bClassificationResponse> {
    return this.http
      .post<B2bClassificationResponse>(`${this.base}/decideur-b2b/classification/predict`, body)
      .pipe(catchError(this.handleError));
  }

  b2bClassifyRisks(body: B2bFeaturesRequest): Observable<B2bClassificationResponse> {
    return this.http
      .post<B2bClassificationResponse>(`${this.base}/decideur-b2b/classification-risks/predict`, body)
      .pipe(catchError(this.handleError));
  }

  b2bCluster(body: B2bFeaturesRequest): Observable<B2bClusteringResponse> {
    return this.http
      .post<B2bClusteringResponse>(`${this.base}/decideur-b2b/clustering/predict`, body)
      .pipe(catchError(this.handleError));
  }

  b2bForecast(body: B2bFeaturesRequest): Observable<B2bRegressionResponse> {
    return this.http
      .post<B2bRegressionResponse>(`${this.base}/decideur-b2b/forecast/predict`, body)
      .pipe(catchError(this.handleError));
  }

  b2bRegress(body: B2bFeaturesRequest): Observable<B2bRegressionResponse> {
    return this.http
      .post<B2bRegressionResponse>(`${this.base}/decideur-b2b/regression/predict`, body)
      .pipe(catchError(this.handleError));
  }

  // FINANCIER
  finCluster(body: FinClusteringRequest): Observable<FinClusteringResponse> {
    return this.http
      .post<FinClusteringResponse>(`${this.base}/decideur-fin/clustering/predict`, body)
      .pipe(catchError(this.handleError));
  }

  finForecast(body: ForecastPeriodRequest): Observable<FinForecastResponse> {
    return this.http
      .post<FinForecastResponse>(`${this.base}/decideur-fin/forecast/predict`, body)
      .pipe(catchError(this.handleError));
  }

  private handleError(error: HttpErrorResponse): Observable<never> {
    let message = 'Erreur inconnue';
    if (error.status === 0) {
      message = 'Impossible de joindre le serveur. Vérifiez que FastAPI tourne sur le port 8000.';
    } else if (error.status === 400) {
      message = `Données invalides : ${error.error?.detail ?? 'vérifiez vos champs'}`;
    } else if (error.status === 503) {
      message = 'Modèle ML non chargé côté serveur.';
    } else {
      message = `Erreur ${error.status} : ${error.error?.detail ?? error.message}`;
    }
    return throwError(() => new Error(message));
  }
}
