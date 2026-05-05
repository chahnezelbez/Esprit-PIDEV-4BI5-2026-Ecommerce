import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { KeycloakService } from 'keycloak-angular';
import { KeycloakRoleService } from '../services/keycloak-role.service';

export function roleGuard(allowedRoles: string[]): CanActivateFn {
  return async () => {
    const keycloak = inject(KeycloakService);
    const roleService = inject(KeycloakRoleService);
    const router = inject(Router);

    const loggedIn = await keycloak.isLoggedIn();
    if (!loggedIn) {
      await keycloak.login();
      return false;
    }

    const userRoles = roleService.getUserRoles();
    const hasRole = allowedRoles.some(role => userRoles.includes(role));

    if (!hasRole) {
      return false;
    }
    return true;
  };
}