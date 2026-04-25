import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { B2bFeaturesRequest, B2bClassificationResponse } from '../../models/api.models';
import { ResultCardComponent } from '../../shared/result-card/result-card.component';

@Component({
  selector: 'app-b2b',
  standalone: true,
  imports: [CommonModule, FormsModule, ResultCardComponent],
  templateUrl: './b2b.component.html',
  styleUrls: ['./b2b.component.scss'],
})
export class B2bComponent {
  loading = signal(false);
  errorMsg = signal<string | null>(null);
  result = signal<B2bClassificationResponse | null>(null);
  featuresValue = '0.15,1,0.4,0.07,0.9,12,220,18,0.02,0.58';

  constructor(private api: ApiService) {}

  get features(): number[] {
    return this.featuresValue
      .split(',')
      .map((value) => Number(value.trim()))
      .filter((value) => !Number.isNaN(value));
  }

  submit(): void {
    this.loading.set(true);
    this.errorMsg.set(null);
    this.result.set(null);

    const payload: B2bFeaturesRequest = { features: this.features };
    this.api.b2bClassify(payload).subscribe({
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

  formatPct(value: number): string {
    return `${(value * 100).toFixed(1)}%`;
  }
}
