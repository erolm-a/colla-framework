<template>
    <router-link :to="url">{{strip(to)}}</router-link>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator'
import {stripPrefix} from '../api';
import KGCollapse from '@/components/KGCollapse.vue';
import { Route } from 'vue-router';

@Component
export default class KGLink extends Vue
{
  @Prop({required: true}) readonly to?: string;

  get url()
  {
    return "/kg/" + this.stripped;
  }

  get stripped()
  {
    if(this.to)
      return stripPrefix(this.to);
    return "";
  }

  private strip(value: string)
  {
    return stripPrefix(value);
  }
};
</script>