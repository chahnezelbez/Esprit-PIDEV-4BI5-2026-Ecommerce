import { Component, Inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { ForecastPeriodRequest, FinForecastResponse } from '../../models/api.models';
import { ResultCardComponent } from '../../shared/result-card/result-card.component';

@Component({
  selector: 'app-financier',
  standalone: true,
  imports: [CommonModule, FormsModule, ResultCardComponent],
  templateUrl: './financier.component.html',
  styleUrls: ['./financier.component.scss'],
})
export class FinancierComponent {
  loading = signal(false);
  errorMsg = signal<string | null>(null);
  result = signal<FinForecastResponse | null>(null);

  request: ForecastPeriodRequest = { periods: 6 };

  constructor(@Inject(ApiService) private api: ApiService) {}

  submit(): void {
    this.loading.set(true);
    this.errorMsg.set(null);
    this.result.set(null);

    this.api.finForecast(this.request).subscribe({
      next: (res) => {
        this.result.set(res);
        this.loading.set(false);
      },
      error: (err: Error) => {
        this.errorMsg.set(err.message);
        this.loading.set(false);
      },
    });
  }
}
