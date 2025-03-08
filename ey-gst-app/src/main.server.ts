import { bootstrapApplication } from '@angular/platform-browser';
import { provideServerRendering } from '@angular/platform-server';
import { AppComponent } from './app/app.component';
import { appConfig } from './app/app.config'; // ✅ Import app-wide configuration

// Bootstrap the Angular SSR application
const bootstrap = () =>
    bootstrapApplication(AppComponent, {
      providers: [
        provideServerRendering(), // ✅ Enables SSR support
        ...appConfig.providers 
      ]
    }).catch(err => console.error(err));
  
  // ✅ Add a default export for compatibility
  export default bootstrap;