
import { Component, OnInit, ViewChild, ElementRef, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MessageComponent } from '../message/message.component';
import { InputFieldComponent } from '../input-field/input-field.component';
import { ChatService } from '../chat/chat.service';
import { Message } from './models';
import { Subscription } from 'rxjs';

@Component({
  selector: 'chat',
  templateUrl: './chat.component.html',
  imports: [MessageComponent, InputFieldComponent,CommonModule],
  styleUrls: ['./chat.component.css']
})
export class ChatComponent implements OnInit, OnDestroy {
  messages: Message[] = [];
  isLoading = false;
  private subscriptions: Subscription[] = [];
  
  @ViewChild('messageContainer') messageContainer!: ElementRef;

  constructor(private chatService: ChatService) {}

  ngOnInit(): void {
    // Subscribe to messages from chat service
    const messageSub = this.chatService.messages$.subscribe(messages => {
      this.messages = messages;
      setTimeout(() => this.scrollToBottom(), 1000);
    });
    
    // Subscribe to loading state
    const loadingSub = this.chatService.loading$.subscribe(isLoading => {
      this.isLoading = isLoading;
    });
    
    this.subscriptions.push(messageSub, loadingSub);
  }

  ngOnDestroy(): void {
    this.subscriptions.forEach(sub => sub.unsubscribe());
  }
  
  /**
   * Handle message sent from the input component
   */
  onChatSendMessage(messageText:string,model:string): void {
    // event.preventDefault(); // Prevent default form behavior (if used in a form)
  
    // const inputElement = event.target as HTMLInputElement;
    // const messageText = inputElement.value.trim();
  
    if (!messageText) return; // Prevent sending empty messages
  
    this.chatService.sendMessage(messageText,model); // Call the service method
  
    //inputElement.value = ''; // Clear input after sending
  }
  
  /**
   * Clear the chat history
   */
  clearChat(): void {
    this.chatService.clearChat();
  }
  
  /**
   * Scroll to the bottom of the message container
   */
  private scrollToBottom(): void {
    if (this.messageContainer) {
      const element = this.messageContainer.nativeElement;
      element.scrollTop = element.scrollHeight;
    }
  }
}