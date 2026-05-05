import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute } from '@angular/router';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { PowerbiConfigService } from '../../services/powerbi-config.service';

@Component({
  selector: 'app-report',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './report.component.html',
  styleUrls: ['./report.component.scss']
})
export class ReportComponent implements OnInit {

  reportUrl: SafeResourceUrl | null = null;
  errorMessage: string | null = null;

  constructor(
    private route: ActivatedRoute,
    private sanitizer: DomSanitizer,
    private powerbiConfig: PowerbiConfigService
  ) {}

  ngOnInit(): void {
    const reportKey = this.route.snapshot.data['reportKey'];
    const config = this.powerbiConfig.getConfig(reportKey);

    if (!config) {
      this.errorMessage = `Aucun rapport configuré pour : ${reportKey}`;
      return;
    }

    this.reportUrl = this.sanitizer.bypassSecurityTrustResourceUrl(config.embedUrl);
  }
}