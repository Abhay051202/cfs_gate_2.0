import { Component, input, OnInit, OnDestroy, signal } from '@angular/core';

@Component({
  selector: 'gate-live-preview', standalone: true,
  styles: [':host { display:block; width:100%; height:100%; } img { width:100%; height:100%; object-fit:contain; }'],
  template: `@if (url()) { <img [src]="url()" [alt]="name()" (load)="next()" (error)="next()"> }`
})
export class LivePreviewComponent implements OnInit, OnDestroy {
  cameraId = input.required<string>();
  name = input('Live camera');
  url = signal('');
  private socket?: WebSocket;
  private timer?: ReturnType<typeof setTimeout>;
  private destroyed = false;
  ngOnInit() { this.connect(); }
  private connect() {
    if (this.destroyed) return;
    const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = this.socket = new WebSocket(`${scheme}//${location.host}/api/cameras/${this.cameraId()}/live`);
    socket.binaryType = 'blob';
    socket.onopen = () => socket.send('next');
    socket.onmessage = event => {
      if (typeof event.data === 'string') { this.next(); return; }
      const previous = this.url();
      this.url.set(URL.createObjectURL(event.data));
      if (previous) URL.revokeObjectURL(previous);
    };
    socket.onclose = () => { if (!this.destroyed) this.timer = setTimeout(() => this.connect(), 1500); };
    socket.onerror = () => socket.close();
  }
  next() {
    clearTimeout(this.timer);
    // Request next frame promptly using requestAnimationFrame to eliminate artificial lag and deliver real-time preview
    if (typeof requestAnimationFrame === 'function') {
      requestAnimationFrame(() => {
        if (this.socket?.readyState === WebSocket.OPEN) this.socket.send('next');
      });
    } else {
      this.timer = setTimeout(() => {
        if (this.socket?.readyState === WebSocket.OPEN) this.socket.send('next');
      }, 10);
    }
  }
  ngOnDestroy() {
    this.destroyed = true;
    clearTimeout(this.timer);
    this.socket?.close();
    if (this.url()) URL.revokeObjectURL(this.url());
  }
}
