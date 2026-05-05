import { Injectable } from '@angular/core';

export interface ReportConfig {
  embedUrl: string;
  reportId: string;
}

@Injectable({ providedIn: 'root' })
export class PowerbiConfigService {

  private readonly BASE_URL = 'https://app.powerbi.com/reportEmbed';
  private readonly REPORT_ID = '84068065-f618-44eb-b89f-4676655167c8';
  private readonly CTID = '604f1a96-cbe8-43f8-abbf-f8eaf5d85730';
  private readonly PARAMS = '&autoAuth=true&filterPaneEnabled=false&navContentPaneEnabled=false';

  private buildUrl(pageName?: string): string {
    let url = `${this.BASE_URL}?reportId=${this.REPORT_ID}&ctid=${this.CTID}${this.PARAMS}`;
    if (pageName) {
      url += `&pageName=${pageName}`;
    }
    return url;
  }

  private readonly reportMap: Record<string, ReportConfig> = {
    achat: {
      reportId: this.REPORT_ID,
      embedUrl: this.buildUrl('3243d4fa598488b71357')   // ← nom de la page Purchases dans Power BI
    },
    vente_b2c: {
      reportId: this.REPORT_ID,
      embedUrl: this.buildUrl('ReportSectionB2C')         // ← nom de la page B2C
    },
    marketing: {
      reportId: this.REPORT_ID,
      embedUrl: this.buildUrl('ReportSectionMarketing')   // ← nom de la page Marketing
    },
    general_manager: {
      reportId: this.REPORT_ID,
      embedUrl: this.buildUrl()                           // ← page par défaut (toutes les pages)
    },
    vente_b2b: {
      reportId: this.REPORT_ID,
      embedUrl: this.buildUrl('ReportSectionB2B')         // ← nom de la page B2B
    },
    financier: {
      reportId: this.REPORT_ID,
      embedUrl: this.buildUrl('ReportSectionFinance')     // ← nom de la page Finance
    }
  };

  getConfig(reportKey: string): ReportConfig | null {
    return this.reportMap[reportKey] ?? null;
  }
}