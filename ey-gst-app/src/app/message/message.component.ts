import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Message } from '../chat/models';

@Component({
  selector: 'app-message',
  templateUrl: './message.component.html',
  imports:[CommonModule],
  styleUrls: ['./message.component.css']
})
export class MessageComponent {
  @Input() message!: Message;
  
  /**
   * Format the timestamp for display
   */
  formatTime(date: Date): string {
    return new Date(date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
}