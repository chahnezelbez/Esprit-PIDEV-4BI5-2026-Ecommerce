import { Component } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  template: `
    <div class="layout">
      <nav class="sidebar">
        <div class="brand">Sougui AI</div>
        <ul>
          <li>
            <a routerLink="/purchase" routerLinkActive="active">
              Purchase
            </a>
          </li>
          <li>
            <a routerLink="/commercial" routerLinkActive="active">
              Commercial
            </a>
          </li>
          <li>
            <a routerLink="/marketing" routerLinkActive="active">
              Marketing
            </a>
          </li>
          <li>
            <a routerLink="/gm" routerLinkActive="active">
              GM
            </a>
          </li>
          <li>
            <a routerLink="/b2b" routerLinkActive="active">
              B2B
            </a>
          </li>
          <li>
            <a routerLink="/financier" routerLinkActive="active">
              Financier
            </a>
          </li>
        </ul>
      </nav>
      <main class="content">
        <router-outlet />
      </main>
    </div>
  `,
  styles: [`
    .layout {
      display: flex;
      min-height: 100vh;
    }
    .sidebar {
      width: 200px;
      min-width: 200px;
      background: #f7f6f2;
      border-right: 0.5px solid #e0dfd8;
      padding: 1.5rem 1rem;
    }
    .brand {
      font-size: 16px;
      font-weight: 500;
      color: #534ab7;
      margin-bottom: 2rem;
      padding-left: 8px;
    }
    ul {
      list-style: none;
      padding: 0;
      margin: 0;
    }
    li { margin-bottom: 4px; }
    a {
      display: block;
      padding: 8px 12px;
      border-radius: 8px;
      text-decoration: none;
      font-size: 14px;
      color: #444441;
      transition: background 0.12s;

      &:hover { background: #eeedfe; color: #534ab7; }
      &.active { background: #eeedfe; color: #534ab7; font-weight: 500; }
    }
    .content {
      flex: 1;
      overflow-y: auto;
      background: #faf9f6;
    }
  `],
})
export class AppComponent {}