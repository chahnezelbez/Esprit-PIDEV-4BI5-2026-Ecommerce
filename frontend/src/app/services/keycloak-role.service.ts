import { Injectable } from '@angular/core';
import { KeycloakService } from 'keycloak-angular';

@Injectable({ providedIn: 'root' })
export class KeycloakRoleService {
  constructor(private keycloak: KeycloakService) {}

  getUserRoles(): string[] {
    const token = this.keycloak.getKeycloakInstance().tokenParsed;
    // Les rôles sont généralement dans realm_access.roles
    return token?.['realm_access']?.['roles'] || [];
  }

  hasRole(role: string): boolean {
    return this.getUserRoles().includes(role);
  }
}