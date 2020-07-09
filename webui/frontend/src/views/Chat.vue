<template>
  <div class="chat">
    <!-- <ChatContainer/> -->
    <BasicVueChat @newOwnMessage="sendMessage"/>
  </div>
</template>

<script lang="ts">
import Vue from 'vue';

import BasicVueChat from 'basic-vue-chat/src/components/basic-vue-chat/BasicVueChat';

export default Vue.extend({
  name: 'Home',
  components: {
    BasicVueChat,
  },

  methods: {
      async sendMessage(message: string) {
          console.log("Written this message: " + message);
          return;
      }
  }
});
</script>
