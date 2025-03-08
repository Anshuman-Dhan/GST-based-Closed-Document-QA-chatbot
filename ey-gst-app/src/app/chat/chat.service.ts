
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, BehaviorSubject } from 'rxjs';
import { Message } from './models';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private messageList: Message[] = [];
  private messageSubject = new BehaviorSubject<Message[]>([]);
  public messages$ = this.messageSubject.asObservable();
  private loading = new BehaviorSubject<boolean>(false);
  public loading$ = this.loading.asObservable();
  public chat_id : string;
  constructor(private http: HttpClient) {
    this.chat_id=this.generateId();
  }

  /**
   * Send a message to the LLM API and handle the response
   * @param text The user's message text
   */
  sendMessage(text: string,model:string=''): void {
    if (!text.trim()) return;
    
    // Add user message to the list
    const userMessage: Message = {
      id: this.generateId(),
      text: text,
      isUser: true,
      timestamp: new Date()
    };
    
    this.addMessage(userMessage);
    
    // Set loading state to true
    this.loading.next(true);
    setTimeout(() => {
      console.log('Delayed action executed!');
    }, 10000); // 2 seconds delay
    // Make API call to the LLM backend
    this.http.post<any>(`${environment.apiUrl}/chat`, {
      query: text,
      model:model,
      context:[],
      chat_id:this.chat_id
    }).subscribe({
      next: (response) => {
        console.log(response);
        // Add bot response to the list
        const botMessage: Message = {
          id: this.generateId(),
          text: response.response || 'Sorry, I could not process your request.',
          isUser: false,
          timestamp: new Date()
        };
        
        this.addMessage(botMessage);
        this.loading.next(false);
      },
      error: (error) => {
        console.error('Error communicating with the chatbot API:', error);
        
        // Add error message
        const errorMessage: Message = {
          id: this.generateId(),
          text: 'Sorry, I encountered an error processing your request. Please try again later.',
          isUser: false,
          timestamp: new Date()
        };
        
        this.addMessage(errorMessage);
        this.loading.next(false);
      }
    });
  }

  /**
   * Clear all messages from the chat
   */
  clearChat(): void {
    this.messageList = [];
    this.messageSubject.next(this.messageList);
  }

  /**
   * Add a new message to the chat
   * @param message The message object to add
   */
  private addMessage(message: Message): void {
    this.messageList = [...this.messageList, message];
    this.messageSubject.next(this.messageList);
  }

  /**
   * Generate a unique ID for messages
   */
  private generateId(): string {
    return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
  }
}