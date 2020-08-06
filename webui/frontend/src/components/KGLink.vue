<template>
    <router-link :to="url">{{stripped}}</router-link>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator'
import {stripPrefix} from '../api';

@Component
export default class KGLink extends Vue
{
  @Prop({required: true}) readonly to?: string;

  get url()
  {
    if(this.to == this.stripped)
      return this.to;
    return "/kg/" + this.stripped;
  }

  get stripped()
  {
    if(this.to)
      return stripPrefix(this.to);
    return "";
  }
};
</script>