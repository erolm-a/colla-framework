<template>
  <div class="chat">
    <!-- <ChatContainer/> -->
    <BasicVueChat title="Grill Chatbot" @newOwnMessage="sendMessage" :new-message="message"/>
  </div>
</template>

<script lang="ts">
import Vue from 'vue';

// @ts-ignore
import BasicVueChat from 'basic-vue-chat/src/components/basic-vue-chat/BasicVueChat';
import { createSession, send, IntentResponse } from '../api';

function getCurrentTime() {
  const currentTime = new Date();
  return `${currentTime.getHours()}:${currentTime.getMinutes()}`
}

class Message {
  id = 1;
  author = "grillbot";
  imageUrl = "";
  image = null;
  contents = "";
  date = "";

  constructor(response: IntentResponse) {
    this.contents = response.message;
    this.date = getCurrentTime();
  }
}

export default Vue.extend({
  name: 'Home',
  components: {
    BasicVueChat,
  },

  async mounted() {
    const response = await createSession();
    this.message = new Message(response);
  },

  data() {
    return {
      message: {}
    }
  },
  
  methods: {
      async sendMessage(message: string) {
          console.log("Written this message: " + message);
          const response = await send(message);
          this.message = new Message(response);
      },
  }
});
</script>
