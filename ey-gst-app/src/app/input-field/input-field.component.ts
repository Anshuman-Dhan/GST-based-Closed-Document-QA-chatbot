import { Component, Output, EventEmitter, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-input-field',
  templateUrl: './input-field.component.html',
  imports: [CommonModule,FormsModule],
  styleUrls: ['./input-field.component.css']
})
export class InputFieldComponent {
  @Output() sendMessageEmitter = new EventEmitter<string>();
  @ViewChild('messageInput') messageInput!: ElementRef;
  
  messageText = '';
  model='';
  
  /**
   * Handle sending the message
   */
  onSendMessage(): void {
    if (this.messageText.trim()) {
      this.sendMessageEmitter.emit(this.messageText);
      this.messageText = '';
    }
  }
  
  /**
   * Handle enter key press to send message
   */
  onKeyDown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.onSendMessage();
    }
  }
}
