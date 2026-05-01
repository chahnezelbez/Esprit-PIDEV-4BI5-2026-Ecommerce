import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';                 // ✅ pour *ngIf
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { KeycloakRoleService } from './services/keycloak-role.service';
import { KeycloakService } from 'keycloak-angular';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterOutlet, RouterLink, RouterLinkActive],
  template: `
    <div class="app-layout">
      <!-- Bouton pour ouvrir/fermer la sidebar -->
      <button class="menu-toggle" (click)="toggleSidebar()" [class.active]="sidebarVisible">
        ☰
      </button>

      <!-- Sidebar avec animation slide -->
      <nav class="sidebar" [class.sidebar--open]="sidebarVisible">
        <div class="brand">
          <div class="logo-container">
            <img src="https://sougui.tn/wp-content/uploads/2021/08/Logo-Sougui-arabe.png.webp"
                 alt="Sougui.tn"
                 class="logo-sougui" />
          </div>
        </div>
        <ul>
          <li *ngIf="roleService.hasRole('achat')">
            <a routerLink="/purchase" routerLinkActive="active" (click)="onNavigate()">
              <span class="icon">🛒</span>
              <span>Décideur Achat</span>
            </a>
          </li>
          <li *ngIf="roleService.hasRole('vente_b2c')">
            <a routerLink="/commercial" routerLinkActive="active" (click)="onNavigate()">
              <span class="icon">📈</span>
              <span>Commercial B2C</span>
            </a>
          </li>
          <li *ngIf="roleService.hasRole('marketing')">
            <a routerLink="/marketing" routerLinkActive="active" (click)="onNavigate()">
              <span class="icon">🎯</span>
              <span>Marketing</span>
            </a>
          </li>
          <li *ngIf="roleService.hasRole('general_manager')">
            <a routerLink="/gm" routerLinkActive="active" (click)="onNavigate()">
              <span class="icon">🏢</span>
              <span>Direction Générale</span>
            </a>
          </li>
          <li *ngIf="roleService.hasRole('vente_b2b')">
            <a routerLink="/b2b" routerLinkActive="active" (click)="onNavigate()">
              <span class="icon">🤝</span>
              <span>B2B</span>
            </a>
          </li>
          <li *ngIf="roleService.hasRole('financier')">
            <a routerLink="/financier" routerLinkActive="active" (click)="onNavigate()">
              <span class="icon">💰</span>
              <span>Finances</span>
            </a>
          </li>
        </ul>
        <div class="logout-btn" (click)="logout()">
          <span class="icon">🔒</span>
          <span>Déconnexion</span>
        </div>
      </nav>

      <main class="content" [class.content--shifted]="sidebarVisible">
        <router-outlet />
      </main>
    </div>
  `,
  styles: [`
    :host {
      --sougui-blue: #1E3A8A;
      --sougui-gold: #D4A017;
      --sougui-orange: #E07A3F;
      --sougui-beige: #F5E9DA;
      --sougui-beige-light: #FAF9F6;
    }

    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    .app-layout {
      display: flex;
      min-height: 100vh;
      background: var(--sougui-beige-light);
      position: relative;
    }

    .menu-toggle {
      position: fixed;
      top: 1rem;
      left: 1rem;
      z-index: 1100;
      background: var(--sougui-gold);
      border: none;
      font-size: 1.6rem;
      padding: 0.4rem 0.8rem;
      border-radius: 12px;
      cursor: pointer;
      color: #1f2937;
      font-weight: bold;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
      transition: all 0.2s;
      line-height: 1;
    }
    .menu-toggle:hover {
      background: #c49b0c;
      transform: scale(1.02);
    }
    .menu-toggle.active {
      left: 290px;
      background: var(--sougui-blue);
      color: white;
    }

    .logo-container {
      background: rgba(255,255,255,0.1);
      border-radius: 60px;
      padding: 6px 18px 6px 14px;
      border: 1px solid rgba(255,255,255,0.2);
      transition: 0.2s;
    }
    .logo-container:hover {
      background: rgba(255,255,255,0.2);
    }
    .logo-sougui {
      height: 48px;
      width: auto;
      display: block;
    }

    .sidebar {
      position: fixed;
      top: 0;
      left: 0;
      width: 280px;
      height: 100vh;
      background: var(--sougui-blue);
      color: white;
      display: flex;
      flex-direction: column;
      padding: 1.5rem 1rem;
      box-shadow: 4px 0 20px rgba(0, 0, 0, 0.2);
      transform: translateX(-100%);
      transition: transform 0.3s cubic-bezier(0.2, 0.9, 0.4, 1.1);
      z-index: 1000;
    }
    .sidebar.sidebar--open {
      transform: translateX(0);
    }

    .sidebar::after {
      content: '◆ ◆ ◆';
      position: absolute;
      bottom: 24px;
      left: 0;
      right: 0;
      text-align: center;
      font-size: 10px;
      color: rgba(255, 255, 255, 0.1);
      letter-spacing: 6px;
      pointer-events: none;
    }

    .brand {
      padding-bottom: 1.8rem;
      margin-bottom: 2rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.15);
    }

    ul {
      list-style: none;
      padding: 0;
      margin: 0;
      flex: 1;
    }
    li {
      margin: 6px 0;
    }
    a {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 10px 16px;
      border-radius: 12px;
      text-decoration: none;
      font-size: 0.9rem;
      font-weight: 500;
      color: rgba(255, 255, 255, 0.8);
      transition: all 0.2s ease;
      cursor: pointer;
    }
    .icon {
      font-size: 1.2rem;
      width: 28px;
      text-align: center;
    }
    a:hover {
      background: rgba(255, 255, 255, 0.1);
      color: white;
      transform: translateX(4px);
    }
    a.active {
      background: var(--sougui-gold);
      color: #1f2937;
      box-shadow: 0 6px 14px rgba(212, 160, 23, 0.3);
      font-weight: 600;
    }

    .logout-btn {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 10px 16px;
      margin-top: 20px;
      border-radius: 12px;
      font-size: 0.9rem;
      font-weight: 500;
      color: rgba(255, 255, 255, 0.8);
      cursor: pointer;
      transition: all 0.2s ease;
      border-top: 1px solid rgba(255,255,255,0.15);
    }
    .logout-btn:hover {
      background: rgba(255, 255, 255, 0.1);
      color: white;
    }

    .content {
      flex: 1;
      overflow-y: auto;
      background: var(--sougui-beige-light);
      transition: margin-left 0.3s ease;
    }
    .content--shifted {
      margin-left: 280px;
    }

    @media (max-width: 768px) {
      .menu-toggle.active {
        left: 1rem;
      }
      .content--shifted {
        margin-left: 0;
      }
      .sidebar {
        width: 100%;
        max-width: 280px;
      }
    }
  `],
})
export class AppComponent {
  sidebarVisible = false;

  constructor(
    public roleService: KeycloakRoleService,
    private keycloak: KeycloakService
  ) {}

  toggleSidebar() {
    this.sidebarVisible = !this.sidebarVisible;
  }

  onNavigate() {
    this.sidebarVisible = false;
  }

  logout() {
    this.keycloak.logout();
  }
}