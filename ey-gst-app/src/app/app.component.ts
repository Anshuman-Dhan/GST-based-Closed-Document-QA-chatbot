import { Component } from '@angular/core';
import { ChatComponent } from './chat/chat.component';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  imports: [ChatComponent,CommonModule],
  standalone: true,
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'ey-gst-app';
}
