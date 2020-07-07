<template>
  <div class="home">
    <SearchTerm/>
  </div>
</template>

<script>
// @ is an alias to /src
// import HelloWorld from '@/components/HelloWorld.vue';
import SearchTerm from '@/components/SearchTerm.vue';

export default {
  name: 'Home',
  components: {
    SearchTerm,
  },
};
</script>
