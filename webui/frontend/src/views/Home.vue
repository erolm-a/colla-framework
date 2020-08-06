<template>
  <div class="home">
    <k-g-search />
  </div>
</template>

<script>
// @ is an alias to /src
// import HelloWorld from '@/components/HelloWorld.vue';
import KGSearch from '@/components/KGSearch.vue';

export default {
  name: 'Home',
  components: {
    KGSearch
  },
};
</script>
