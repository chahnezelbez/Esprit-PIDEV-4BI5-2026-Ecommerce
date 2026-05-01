import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'maxYhat',
  standalone: true,
})
export class MaxYhatPipe implements PipeTransform {
  transform(predictions: { yhat: number }[] | undefined | null): number {
    if (!predictions || predictions.length === 0) {
      return 0;
    }
    return Math.max(...predictions.map(p => p.yhat));
  }
}