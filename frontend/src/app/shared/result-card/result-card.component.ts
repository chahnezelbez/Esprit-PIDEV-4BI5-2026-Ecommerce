import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-result-card',
  standalone: true,
  imports: [CommonModule],
  template: `
    <section class="result-card">
      <header *ngIf="title"><h2>{{ title }}</h2></header>
      <div class="result-content">
        <ng-content></ng-content>
      </div>
    </section>
  `,
  styles: [`
    .result-card {
      background: #ffffff;
      border: 1px solid #e5e4e1;
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 14px 34px rgba(0, 0, 0, 0.04);
    }
    .result-card header {
      margin-bottom: 1rem;
    }
    .result-card h2 {
      margin: 0;
      font-size: 1rem;
      font-weight: 600;
      color: #2f2d45;
    }
    .result-content {
      color: #4b4a56;
      font-size: 0.95rem;
      line-height: 1.6;
    }
  `],
})
export class ResultCardComponent {
  @Input() title = '';
}
